# Regime Alpha Benchmark

> A research-grade quant ML benchmark platform that tests whether learned market regimes improve alpha modeling versus global baselines and classical regime methods, using proper financial labels, purged walk-forward validation, and transaction-cost-aware evaluation.

**Maintainer:** [sakshibhongle1221](https://github.com/sakshibhongle1221)

**Run locally:** `python -m streamlit run streamlit_app.py`. See the setup instructions below.

**Deploy:** Connect your own GitHub repository to Streamlit Community Cloud and select `streamlit_app.py` as the entry point.

## Current Research Finding After Phase 49

Phase 49 is the current final submission-closure checkpoint. Phase 39R repaired a serious validation weakness in the earlier Crypto-20 fold-local experiment: the original per-symbol positional folds overlapped in calendar time when assets were pooled. That earlier Phase 39 run is therefore kept only as audit history, not as predictive evidence.

Under the repaired `crypto20-development-v1` protocol, every method is evaluated on the same globally separated calendar folds, with a frozen development dataset, fold-local fitting, equal coverage, mean per-asset IC as the primary ranking metric, and a non-overlapping transaction-cost-aware portfolio diagnostic.

The honest Phase 39R/40 result is:

- the repaired classical baselines are weak/negative;
- the repaired neural/guided methods do not show convincing positive alpha;
- HMM-guided and contrastive methods do not currently establish robust dominance over the simpler repaired baselines;
- Phase 40 statistical adjudication finds no corrected IC/Sharpe superiority claim;
- the project’s strongest contribution is now the research-grade benchmark, leakage repair, validation discipline, and clear separation between structural regime quality and tradable alpha.

This makes the project more scientifically defensible, not less. The current claim is no longer “our model beats HMM.” The current claim is: **strict financial validation exposes that structural regime learning can look meaningful while downstream alpha remains weak, noisy, and method-sensitive.**

Phase 41 registers bounded calibration, soft-gating, and execution-control candidates. Phase 41B then runs the first full inner-validation-selected global/classical candidate experiment for the probability-calibration and soft-gating subset only; score-threshold execution control is registered but deferred. It is a controlled negative/diagnostic result: candidate selection stays inside each outer fold, but the run does not improve the repaired alpha conclusion or support corrected IC/Sharpe dominance.

Phase 43B completes the one-shot locked external holdout. The frozen final candidate, `regime_lgbm_hmm_guided_hmm`, satisfies the prewritten relative IC/Sharpe rule versus `global_lgbm` and `regime_lgbm_hmm`, but its locked Sharpe and total return are still negative. Therefore the paper may claim limited locked relative support, but it must not claim a profitable, deployable, or broadly dominant trading strategy.

Phase 44 converts this into a paper-facing evidence package, reviewer brief, and submission risk register. Phase 45 through Phase 49 then harden the manuscript, submission package, compiled PDF, claim traceability, reviewer-reading guide, final validation-repair figure, final system-overview figure, citation coverage, PDF metadata audit, and artifact-release boundary. These are packaging and claim-control phases, not model-rescue phases.

## How To Read The Historical Phase Notes

The phase-by-phase sections below are a chronological engineering and research log. Some older sections describe positive-looking development results from before the Phase 39R repair and Phase 43B locked holdout. Those sections are preserved for audit history, but they are not the current paper claim.

For the current scientific interpretation, prefer:

- `paper/main.md`
- `reports/phase44_paper_readiness_package.md`
- `reports/phase44_reviewer_brief.md`
- `reports/claim_registry.md`
- `reports/phase49_final_submission_closure.md`
- `paper/phase49_submission_ready_manuscript.tex`
- `models/research_grade_check_report.md`

## Research Question

Does learning market regimes from raw financial time-series features improve alpha-model IC, drawdown, and Sharpe compared with:

- a global model with no regime awareness
- Gaussian HMM regimes
- contrastive-HMM hybrid regimes
- HMM-guided learned regimes
- KMeans regimes
- volatility-bucket regimes

The project is intentionally framed as a research benchmark, not a live trading bot. Honest weak or mixed results are part of the contribution.

## Research Positioning

Regime Alpha Benchmark sits between four research areas:

- contrastive representation learning for time series, such as TS2Vec, TNC, CoST, TS-TCC, and TF-C
- classical financial regime-switching models, including Hamilton-style Markov switching, Gaussian HMMs, and Markov-switching GARCH
- financial ML validation discipline: triple-barrier labels, purging, embargoes, fold-level testing, and multiple-testing control
- regime-conditioned alpha modeling, where predictors are trained or weighted differently across market states

The contribution is not "a new trading bot" or "HMM is ground truth." The contribution is a reproducible benchmark that tests whether learned regime embeddings can beat, match, or explain classical sequential regimes under financial validation. Phase 18 adds the first model-side response to the current finding: an HMM-guided contrastive encoder that uses HMM states as weak supervision rather than hard truth.

The full positioning note is in `reports/related_work.md`, with a compact source matrix in `reports/literature_matrix.csv`.

## Architecture

```text
Binance OHLCV
    -> DuckDB feature store
    -> 22 engineered market/microstructure features
    -> Multi-horizon financial labels and triple-barrier targets
    -> Contrastive temporal regime encoder + contrastive-HMM hybrid + classical baselines
    -> Global and regime-conditioned LightGBM models
    -> Purged walk-forward validation with transaction costs
    -> Research dashboard and report artifacts
```

## Current Capabilities

- Incremental Binance OHLCV ingestion for BTCUSDT and ETHUSDT.
- DuckDB feature store with 22 engineered features.
- Contrastive Transformer-style temporal encoder trained with NT-Xent loss.
- GMM regime posteriors and UMAP/timeline visualizations.
- Multi-horizon targets: 4h, 8h, and 24h.
- Triple-barrier labels with neutral class.
- Regime baselines: contrastive, Gaussian HMM, KMeans, volatility buckets.
- Contrastive-HMM hybrid regimes fitted on learned embeddings.
- Regime stability diagnostics: switch rate, duration, transition entropy, and stable-vs-transition IC.
- Phase 16 regime-quality diagnostics: balance, persistence, posterior confidence, pairwise NMI/ARI, and HMM-reference agreement.
- Global LightGBM baseline and regime-conditioned LightGBM models.
- 5-day embargoed walk-forward validation.
- Validation audit for fold separation, target horizon leakage, coverage parity, and prediction alignment.
- Fold-local regime refitting benchmark for stricter predictive regime evaluation.
- Phase 14A robustness matrix across BTC-only, ETH-only, and BTC+ETH scopes at 4h, 8h, and 24h horizons.
- Phase 14B stress robustness across transaction costs, signal thresholds, and bull/sideways/bear market periods.
- Phase 15.0 run registry for timestamped artifact snapshots and frozen baselines.
- Phase 15A statistical testing with fold-level bootstrap confidence intervals, paired tests, and DM-style forecast-loss checks.
- Phase 15B multiple-testing correction, corrected claim status, and Probabilistic Sharpe diagnostics.
- Phase 17 compute planning with encoder timing, ablation budget, and experiment-priority queue.
- Phase 18 HMM-guided contrastive encoder prototype with weak HMM-state positives and boundary-aware hard negatives.
- Phase 19A literature positioning across time-series contrastive learning, financial regime switching, financial ML validation, and regime-conditioned alpha modeling.
- Phase 19B full 30-epoch HMM-guided encoder run with guided-vs-baseline structural comparison.
- Phase 20 guided downstream alpha retest with fold-local GMM/HMM assignment layers on the guided embedding space.
- Phase 22A time-frequency guided encoder prototype with FFT magnitude bands appended to each time window.
- Phase 23 fold-local LightGBM feature-importance and SHAP diagnostics for paper interpretability.
- Phase 24 paper protocol freeze with hypotheses, claim boundaries, and experiment manifest.
- Phase 25 minimal ablation suite for objective guidance, assignment layer, augmentation view, and classical-reference comparisons.
- Phase 26 paper-facing statistical claim tests.
- Phase 27 generated manuscript skeleton, artifact map, and submission checklist.
- Phase 28 reproducibility package with smoke/full/dashboard reproduction modes and artifact policy docs.
- Phase 29 paper prose pass that turns the manuscript scaffold into submission-style Markdown.
- Phase 30 reviewer-defense framing for likely objections around scope, fold power, and statistical claims.
- Phase 31 pre-specified Crypto-20/Crypto-50 universe protocol.
- Phase 32 Crypto-20 ingestion, feature, target, and quality-gate pipeline.
- Phase 33 Crypto-20 classical regime benchmark.
- Phase 34 Crypto-20 guided-encoder readiness gate before expensive learned-regime training.
- Phase 35 full Crypto-20 HMM-guided encoder structural generalization run.
- Phase 36 Crypto-20 fold-local downstream alpha retest.
- Phase 37 paired-fold statistical adjudication, time-block DM diagnostics, and asset heterogeneity analysis.
- Phase 38 research-control reset with explicit data roles, experiment lineage, fold-local validity requirements, and publication acceptance gates.
- Phase 39 fold-local benchmark engineering plus a calendar-leakage repair: the original positional Crypto-20 run is retained as invalidated development evidence, while the repaired pipeline enforces one shared timestamp index and strict pooled train/test separation.
- Repaired evaluation protocol with mean per-asset IC as the primary ranking metric, separate cross-sectional and pooled IC diagnostics, and a pre-specified non-overlapping eight-hour portfolio grid with fold-boundary position resets.
- Completed repaired Crypto-20 classical and fold-local neural/guided development benchmarks. Under the repaired protocol, downstream alpha remains weak/inconclusive rather than a positive trading result.
- Completed Phase 43B locked external holdout: limited prewritten relative support is satisfied, but tradable positive alpha is not supported.
- Phase 44 paper-readiness package with evidence matrix, reviewer brief, submission risk register, and updated manuscript draft.
- Research-grade regression gate for future feature changes: artifact checks validate frozen data, method coverage, completed checkpoints, repaired outputs, and claim-control docs; full checks additionally rerun freeze verification, unit tests, and the calendar audit.
- Transaction-cost-aware experiment result table.
- Streamlit dashboard shell and research note.

## Research-Grade Check Loop

Run this after every new feature or experiment-script change:

```powershell
.\run_research_grade_checks.ps1
```

Run the stronger gate before committing or interpreting results:

```powershell
.\run_research_grade_checks.ps1 -Mode full
```

The full gate checks the frozen development dataset, repaired artifact schemas, equal method coverage, 16 completed neural checkpoints, claim-control wording, unit tests, and common-calendar train/test separation.

## Run Order

Recommended local environment:

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements-research.txt
```

```powershell
python src/check.py
python src/multiasset_universe.py
python src/targets.py --symbols BTCUSDT ETHUSDT
python src/train_encoder.py --symbols BTCUSDT ETHUSDT
python src/visualize_regimes.py --symbols BTCUSDT ETHUSDT
python src/baselines.py --symbols BTCUSDT ETHUSDT
python src/alpha_models.py --symbols BTCUSDT ETHUSDT
python src/regime_stability.py --symbols BTCUSDT ETHUSDT
python src/regime_quality.py --symbols BTCUSDT ETHUSDT
python src/compute_plan.py --symbols BTCUSDT ETHUSDT
python src/guided_encoder.py --symbols BTCUSDT ETHUSDT --epochs 30
python src/guided_encoder.py --symbols BTCUSDT ETHUSDT --augmentation time_frequency --epochs 3
python src/walkforward_regimes.py --symbols BTCUSDT ETHUSDT
python src/robustness.py
python src/robustness_stress.py
python src/statistical_tests.py
python src/interpretability.py --symbols BTCUSDT ETHUSDT
python src/ablation_suite.py
python src/paper_claim_tests.py
python src/paper_skeleton.py
python src/validation_audit.py --symbols BTCUSDT ETHUSDT
.\reproduce.ps1 -Mode smoke
python src/archive_run.py --phase phase14b_baseline --run-id 20260522_phase14b_baseline --source-ref v1.3-phase14b --notes "Frozen Phase 14B baseline before Phase 15 statistical and encoder work."
python src/backtest.py
python -m compileall src dashboard.py streamlit_app.py
```

Crypto-20 data-pipeline pilot, run only when you are ready to download the expanded asset history:

```powershell
python src/multiasset_universe.py
python src/ingestion.py --universe crypto20
python src/features.py --universe crypto20
python src/targets.py --universe crypto20 --artifact-prefix crypto20_
python src/check.py --universe crypto20
python src/crypto20_quality_gate.py --universe crypto20
python src/crypto20_regime_benchmark.py --universe crypto20
python src/crypto20_guided_readiness.py --universe crypto20
.\scripts\phases\run_phase35_crypto20_guided.ps1
.\scripts\phases\run_phase36_crypto20_alpha.ps1
.\scripts\phases\run_phase37_crypto20_statistics.ps1
python src/freeze_development_dataset.py --verify-only
.\scripts\phases\run_phase39r_classical_baseline.ps1 -MaxFolds 1 -RunName phase39r_classical_smoke -OutputDir .tmp\phase39r_classical_smoke -OutputPrefix smoke_
```

Optional dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

For Streamlit Cloud deployment, the root `requirements.txt` is intentionally minimal and installs only dashboard dependencies.

If local research dependencies are not installed, run:

```powershell
python -m pip install -r requirements-research.txt
```

## Key Artifacts

| Artifact | Purpose |
|---|---|
| `models/target_distribution.csv` | Label balance across horizons and label types |
| `models/target_quality.csv` | Missing-row, horizon-loss, and neutral-class checks |
| `models/asset_universe_crypto20.csv` | Phase 31 pre-specified Crypto-20 expansion universe |
| `models/asset_universe_crypto50.csv` | Phase 31 pre-specified Crypto-50 expansion universe |
| `models/asset_universe_exclusions.csv` | Phase 31 transparent exclusion and pending-ingestion log |
| `reports/multiasset_universe_plan.md` | Phase 31 reviewer-facing multi-asset protocol |
| `reports/crypto20_pipeline_plan.md` | Phase 32 reproducible Crypto-20 ingestion and quality-gate plan |
| `models/crypto20_data_quality.csv` | Phase 32 per-symbol Crypto-20 OHLCV/feature/target quality gate |
| `models/crypto20_pipeline_summary.csv` | Phase 32 compact Crypto-20 pass-rate and gate-status summary |
| `models/crypto20_target_distribution.csv` | Phase 32 Crypto-20 label distribution diagnostics |
| `models/crypto20_target_quality.csv` | Phase 32 Crypto-20 target horizon-loss and class-balance checks |
| `models/crypto20_regime_benchmark_summary.csv` | Phase 33 Crypto-20 classical regime benchmark summary |
| `models/crypto20_per_regime_stats.csv` | Phase 33 per-regime forward-return, volatility, liquidity, and IC diagnostics |
| `models/crypto20_regime_symbol_summary.csv` | Phase 33 per-symbol regime persistence and balance diagnostics |
| `reports/crypto20_regime_benchmark_plan.md` | Phase 33 reviewer-facing multi-asset regime benchmark protocol |
| `models/crypto20_guided_pair_summary.csv` | Phase 34 HMM-guided pair-mining readiness diagnostics for Crypto-20 |
| `models/crypto20_guided_compute_plan.csv` | Phase 34 compute gate for the full Crypto-20 guided encoder run |
| `models/crypto20_guided_gate.csv` | Phase 34 go/no-go recommendation for learned-regime expansion |
| `reports/crypto20_guided_readiness.md` | Phase 34 reviewer-facing readiness note before expensive guided training |
| `models/crypto20_guided_encoder_summary.csv` | Phase 35 full Crypto-20 guided encoder structural summary, after the long run is executed |
| `models/crypto20_guided_encoder_loss.csv` | Phase 35 training loss and pair-mining diagnostics |
| `models/crypto20_guided_encoder_comparison.csv` | Phase 35 guided-regime comparison against the Phase 33 Crypto-20 classical baseline |
| `reports/crypto20_alpha_generalization.md` | Phase 36 protocol for the Crypto-20 downstream alpha retest |
| `models/crypto20_walkforward_experiment_results.csv` | Phase 36 Crypto-20 fold-local alpha benchmark summary |
| `models/crypto20_walkforward_regime_summary.csv` | Phase 36 fold-local Crypto-20 regime coverage and persistence diagnostics |
| `models/crypto20_walkforward_guided_alpha_comparison.csv` | Phase 36 guided-vs-classical downstream alpha comparison |
| `models/crypto20_walkforward_equity_curve.png` | Phase 36 Crypto-20 fold-local equity curve visualization |
| `reports/crypto20_statistical_protocol.md` | Phase 37 pre-specified statistical protocol and completed findings |
| `models/crypto20_statistical_method_summary.csv` | Phase 37 fold-level method estimates and bootstrap intervals |
| `models/crypto20_statistical_claims.csv` | Phase 37 multiple-testing-aware claim statuses |
| `models/crypto20_statistical_asset_metrics.csv` | Secondary per-asset heterogeneity diagnostics |
| `models/crypto20_statistical_ic_confidence_intervals.png` | Crypto-20 mean fold IC confidence intervals |
| `reports/phase38_master_protocol.md` | Phase 38 scientific scope, data-use, fold-local, and model-selection protocol |
| `reports/data_role_registry.csv` | Paper-facing registry of development-observed, descriptive, locked, and future datasets |
| `configs/crypto20_development_freeze_v1.json` | Versioned asset, timestamp, target, fold, and role contract for the repaired development benchmark |
| `models/crypto20_development_freeze_manifest.json` | Database, experiment-data, universe, symbol-manifest, and fold-calendar hashes |
| `models/crypto20_development_symbol_manifest.csv` | Per-symbol frozen row counts, gaps, timestamps, and hashes |
| `models/crypto20_development_fold_calendar.csv` | All 16 globally separated calendar folds |
| `reports/crypto20_development_data_freeze.md` | Honest provenance and integrity statement for the development snapshot |
| `reports/phase39r_classical_baseline_protocol.md` | Frozen four-method gate before repaired neural retraining |
| `scripts/phases/run_phase39r_classical_baseline.ps1` | Resume-safe PowerShell runner for the repaired classical benchmark |
| `reports/experiment_ledger.csv` | Complete inspected/planned experiment-family ledger |
| `reports/publication_acceptance_gates.md` | Ordered scientific, paper, and BTech project completion gates |
| `reports/phase39_fold_local_encoder_design.md` | Code-grounded implementation contract for the fully fold-local learned baseline |
| `reports/phase39_fold_local_results.md` | Invalidated original Phase 39 history retained only for audit |
| `reports/phase39r_neural_fold_local_results.md` | Completed repaired 16-fold fold-local neural/guided development benchmark |
| `models/crypto20_repaired_fold_local_experiment_results.csv` | Repaired neural/guided method summary under common-calendar validation |
| `reports/phase40_repaired_statistical_adjudication.md` | Phase 40 repaired statistical adjudication and paper-safe interpretation |
| `models/crypto20_repaired_fold_local_statistical_method_summary.csv` | Phase 40 repaired fold-level statistical summary |
| `models/crypto20_repaired_fold_local_statistical_claims.csv` | Phase 40 corrected claim statuses for repaired outputs |
| `reports/phase41_bounded_improvement_protocol.md` | Phase 41 bounded calibration and soft-gating protocol |
| `models/phase41_candidate_registry.csv` | Registered Phase 41 candidate families, including deferred score-threshold execution control |
| `models/phase41_selection_rules.csv` | Mandatory inner-validation selection and guardrail rules |
| `src/phase41_inner_validation_candidates.py` | Phase 41B inner-validation candidate runner for the global/classical probability-calibration/soft-gating ladder |
| `reports/phase41_inner_validation_candidate_run.md` | Phase 41B full global/classical candidate findings |
| `models/phase41_classical_experiment_results.csv` | Phase 41B full global/classical candidate method summary |
| `models/phase41_classical_statistical_claims.csv` | Phase 41B corrected statistical claim statuses |
| `reports/phase42_interpretation_execution_hardening.md` | Phase 42 explanation of weak repaired alpha via execution, transition, cross-asset, and feature diagnostics |
| `models/phase42_execution_stress_summary.csv` | Phase 42 transaction-cost and score-threshold sensitivity summary |
| `models/phase42_regime_transition_diagnostics.csv` | Phase 42 regime switch-rate, duration, confidence, and balance diagnostics |
| `models/phase42_cross_asset_alpha_diagnostics.csv` | Phase 42 per-asset alpha fragility summary |
| `models/phase42_feature_family_diagnostics.csv` | Phase 42 feature-family target-alignment diagnostics |
| `reports/phase43_locked_holdout_freeze.md` | Phase 43A final candidate, holdout-rule, and claim-rule freeze before locked evaluation |
| `models/phase43_locked_candidate_manifest.csv` | Frozen final guided-HMM candidate, references, exclusions, and support-artifact hashes |
| `models/phase43_locked_claim_rules.csv` | Locked-holdout success, failure, allowed-claim, and forbidden-claim rules |
| `models/phase43_locked_holdout_rules.csv` | External/future holdout selection and no-retuning rules |
| `reports/phase43b_locked_holdout_registration.md` | Phase 43B locked-holdout registration/readiness gate before model evaluation |
| `models/phase43b_locked_holdout_registration_manifest.csv` | Registration status, selected-count, symbol hash, and forbidden-input confirmation |
| `models/phase43b_holdout_candidate_quality.csv` | Pre-outcome quality table for candidate external holdout assets |
| `models/phase43b_registered_holdout_symbols.csv` | Registered Phase 43B external locked-holdout symbols |
| `reports/phase43b_locked_holdout_data_freeze.md` | Frozen locked-holdout data hash, symbols, and fold calendar |
| `reports/phase43b_locked_external_evaluation.md` | One-shot locked external evaluation report |
| `reports/phase43b_locked_external_adjudication.md` | Paper-safe locked result interpretation and forbidden claims |
| `models/phase43b_locked_external_experiment_results.csv` | Locked external method summary across 18 folds |
| `models/phase43b_locked_external_primary_comparison.csv` | Frozen final candidate versus primary references under the Phase 43A rule |
| `models/phase43b_locked_external_claims.csv` | Locked claim adjudication, including no tradable-positive claim |
| `reports/phase44_paper_readiness_package.md` | Paper-facing evidence package after the locked holdout |
| `reports/phase44_reviewer_brief.md` | Short reviewer-response brief for claim boundaries and likely objections |
| `reports/phase44_prepush_hardening_audit.md` | Final pre-push audit of stale claims, reviewer risks, and Phase 44 guardrails |
| `models/phase44_paper_evidence_matrix.csv` | Reviewer-facing map of evidence blocks, data roles, findings, and claim boundaries |
| `models/phase44_submission_risk_register.csv` | Submission risk register for overclaiming, candidate switching, leakage, and venue packaging |
| `models/research_grade_check_report.csv` | Latest automated artifact/full research-grade check result |
| `models/regime_assignments.csv` | Aligned regime labels/posteriors for all methods |
| `models/regime_benchmark_summary.csv` | Regime-level comparison table |
| `models/regime_stability_summary.csv` | Persistence, switch-rate, confidence, and stable-vs-transition IC diagnostics |
| `models/regime_quality_summary.csv` | Phase 16 structural regime-quality metrics independent of alpha performance |
| `models/regime_agreement_matrix.csv` | Pairwise NMI/ARI/same-label agreement across regime methods |
| `models/regime_quality_heatmap.png` | Visual summary of balance, persistence, confidence, and HMM-reference agreement |
| `models/regime_agreement_heatmap.png` | Pairwise method-agreement heatmap |
| `models/compute_profile.csv` | Phase 17 local encoder timing and train-cost estimate |
| `models/ablation_budget.csv` | Prioritized 12-run encoder ablation queue |
| `models/compute_budget_summary.csv` | Compact compute-budget summary for dashboard/report use |
| `models/compute_budget_plan.png` | Visual ablation-runtime budget |
| `models/ablation_results.csv` | Phase 25 metric-level ablation comparisons |
| `models/ablation_summary.csv` | Phase 25 paper-facing ablation decision table |
| `models/ablation_heatmap.png` | Visual summary of Phase 25 ablation win rates |
| `models/guided_encoder_summary.csv` | Phase 19B HMM-guided encoder structural diagnostics |
| `models/guided_encoder_loss.csv` | Phase 19B guided training loss and pair-mining diagnostics |
| `models/guided_encoder_comparison.csv` | Phase 19B comparison of guided encoder regimes versus existing structural baselines |
| `models/guided_alpha_comparison.csv` | Phase 20 downstream comparison of guided regime alpha models versus global/classical references |
| `models/guided_encoder_loss_curve.png` | Visual Phase 19B training loss curve |
| `models/time_frequency_encoder_summary.csv` | Phase 22A time-frequency guided encoder structural diagnostics |
| `models/time_frequency_encoder_loss.csv` | Phase 22A time-frequency guided training loss diagnostics |
| `models/time_frequency_encoder_comparison.csv` | Phase 22A comparison against baseline regime-quality methods |
| `models/time_frequency_encoder_loss_curve.png` | Visual Phase 22A time-frequency training loss curve |
| `models/feature_importance_global.csv` | Phase 23 fold-local global LightGBM importance summary |
| `models/feature_importance_by_regime.csv` | Phase 23 fold-local raw-HMM and guided-HMM regime-conditioned importance summary |
| `models/feature_family_summary.csv` | Phase 23 feature-family attribution summary for paper interpretation |
| `models/feature_importance_by_regime.png` | Visual Phase 23 top features by regime-conditioned model |
| `models/feature_family_importance.png` | Visual Phase 23 feature-family attribution by regime |
| `models/per_regime_stats.csv` | Volatility, return, liquidity, and IC diagnostics by regime |
| `models/experiment_results.csv` | Master model comparison table |
| `models/alpha_oos_predictions.csv` | Out-of-sample model predictions |
| `models/validation_audit.csv` | Leakage, embargo, fold, coverage, and prediction-alignment audit |
| `models/fold_audit.csv` | Fold-level train/test boundary and embargo checks |
| `models/walkforward_experiment_results.csv` | Strict fold-local regime-refit alpha benchmark |
| `models/walkforward_comparison.csv` | Offline/global versus fold-local regime result comparison |
| `models/walkforward_regime_summary.csv` | Fold-local regime coverage, duration, and confidence diagnostics |
| `models/robustness_results.csv` | Phase 14A full horizon/symbol-scope robustness matrix |
| `models/robustness_summary.csv` | Best method per robustness grid cell |
| `models/robustness_wins.csv` | Win counts by metric across robustness grid cells |
| `models/robustness_heatmap.png` | Visual summary of robustness winners |
| `models/robustness_stress_results.csv` | Phase 14B full cost/threshold/market-period stress grid |
| `models/robustness_stress_summary.csv` | Best method per stress grid cell |
| `models/robustness_stress_wins.csv` | Stress-test win counts by metric |
| `models/robustness_stress_heatmap.png` | Visual summary of stress-test sensitivity |
| `models/statistical_method_summary.csv` | Phase 15A fold-level confidence intervals by method |
| `models/statistical_pairwise_tests.csv` | Phase 15A paired and DM-style method comparisons |
| `models/statistical_test_summary.csv` | Compact paper-facing significance summary |
| `models/statistical_ic_confidence_intervals.png` | Fold-level IC confidence interval plot |
| `models/statistical_multiple_testing.csv` | Phase 15B corrected p-values across tested claims |
| `models/statistical_claims.csv` | Phase 15B corrected claim-status summary |
| `models/statistical_sharpe_diagnostics.csv` | Probabilistic Sharpe Ratio diagnostics |
| `models/paper_claim_tests.csv` | Phase 26 metric-level paper claim tests mapped to the ablation suite |
| `models/paper_statistical_summary.csv` | Phase 26 paper-facing statistical claim summary |
| `models/paper_claim_tests.png` | Visual Phase 26 paper claim status summary |
| `paper/main.md` | Phase 44 paper-readiness manuscript draft after repaired and locked evidence |
| `reports/paper_artifact_map.csv` | Phase 44 paper section to artifact map |
| `reports/paper_submission_checklist.md` | Phase 44 submission-readiness checklist |
| `models/statistical_multiple_testing.png` | Visual multiple-testing correction summary |
| `models/statistical_sharpe_diagnostics.png` | Visual PSR diagnostic summary |
| `runs/run_index.csv` | Versioned run registry |
| `runs/20260522_phase14b_baseline/manifest.json` | Frozen Phase 14B baseline manifest |
| `reproduce.ps1` | Phase 28 PowerShell reproduction helper |
| `reproduce.sh` | POSIX shell reproduction helper for Linux/macOS reviewers |
| `scripts/phases/run_phase35_crypto20_guided.ps1` | PowerShell runner for the long Crypto-20 guided encoder experiment |
| `scripts/phases/run_phase35_crypto20_guided.sh` | POSIX shell runner for the long Crypto-20 guided encoder experiment |
| `reports/environment.md` | Phase 28 local and deployment environment notes |
| `reports/artifact_manifest.md` | Phase 28 committed/regenerated/ignored artifact policy |
| `reports/reproduction_checklist.md` | Phase 28 reviewer reproduction checklist |
| `reports/model_card.md` | Reproducible model-card snapshot |
| `reports/compute_budget.md` | Compute-aware experiment plan and multi-asset gate |
| `reports/related_work.md` | Phase 19A paper-positioning note and contribution map |
| `reports/literature_matrix.csv` | Compact source matrix for paper planning |
| `reports/paper_protocol.md` | Phase 24 frozen research question, method scope, metrics, and decision gates |
| `reports/hypotheses.md` | Phase 24 hypothesis table and current evidence status |
| `reports/claim_registry.md` | Phase 24 allowed, directional, open, and forbidden claim language |
| `reports/experiment_manifest.md` | Phase 24 completed experiments, future queue, and submission-readiness checklist |
| `reports/paper_artifact_map.csv` | Phase 27 mapping from manuscript sections to evidence artifacts |
| `reports/paper_submission_checklist.md` | Phase 27 checklist for paper readiness and forbidden claims |
| `models/regime_stability.png` | Stability and transition-period IC dashboard panel |
| `models/phase4_dashboard.png` | Static research backtest dashboard |
| `reports/regime_alpha_benchmark_report.md` | Research note |

## Project Structure

```text
regime-alpha-benchmark/
├── dashboard.py
├── streamlit_app.py
├── reproduce.ps1
├── reproduce.sh
├── scripts/
│   └── phases/
│       ├── run_phase35_crypto20_guided.ps1
│       ├── run_phase35_crypto20_guided.sh
│       ├── run_phase36_crypto20_alpha.ps1
│       ├── run_phase36_crypto20_alpha.sh
│       ├── run_phase37_crypto20_statistics.ps1
│       └── ...
├── reports/
│   ├── regime_alpha_benchmark_report.md
│   ├── related_work.md
│   ├── literature_matrix.csv
│   ├── model_card.md
│   ├── compute_budget.md
│   ├── paper_protocol.md
│   ├── hypotheses.md
│   ├── claim_registry.md
│   ├── experiment_manifest.md
│   ├── paper_artifact_map.csv
│   ├── paper_submission_checklist.md
│   ├── environment.md
│   ├── artifact_manifest.md
│   └── reproduction_checklist.md
├── paper/
│   └── main.md
├── src/
│   ├── ingestion.py
│   ├── features.py
│   ├── targets.py
│   ├── dataset.py
│   ├── encoder.py
│   ├── train_encoder.py
│   ├── visualize_regimes.py
│   ├── baselines.py
│   ├── regime_stability.py
│   ├── regime_quality.py
│   ├── compute_plan.py
│   ├── ablation_suite.py
│   ├── guided_encoder.py
│   ├── interpretability.py
│   ├── validation_audit.py
│   ├── walkforward_regimes.py
│   ├── robustness.py
│   ├── robustness_stress.py
│   ├── statistical_tests.py
│   ├── paper_claim_tests.py
│   ├── paper_skeleton.py
│   ├── archive_run.py
│   ├── alpha_models.py
│   ├── backtest.py
│   └── check.py
├── models/
│   └── generated research artifacts
└── runs/
    └── versioned curated run snapshots
```

## Methodology Notes

The primary target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The alpha score is `P(+1) - P(-1)`. A prediction becomes a trade only when the neutral class is not dominant and the absolute score exceeds the threshold.

The validation scheme uses expanding walk-forward folds with a 5-day embargo gap between train and test windows. This reduces leakage from overlapping financial labels and makes the model comparison more defensible.

Phase 12 adds a validation audit. The current audit passes all critical checks: required tables, feature/target schema, finite joined rows, 24-row target horizon loss, 18 walk-forward folds, 120-bar embargo, 8-bar label-horizon purge, equal method coverage, prediction/test-fold alignment, experiment-result row counts, Phase 20 fold-local guided-alpha artifact coverage, Phase 14A robustness artifact completeness, Phase 14B stress-grid completeness, Phase 15A/15B statistical artifact completeness, Phase 16 regime-quality artifact completeness, Phase 17 compute-plan artifact completeness, Phase 19B guided-encoder full-run artifact completeness, Phase 25 ablation artifact completeness, Phase 19A literature-positioning artifact completeness, and run-registry snapshot completeness.

The audit also records one methodological warning: the legacy `regime_assignments.csv` artifact is offline/global. Paper-grade predictive regime claims should use the Phase 13 `walkforward_experiment_results.csv` artifact instead.

Phase 13 adds that stricter benchmark. In `walkforward_regimes.py`, the regime assignment layer is refit inside each fold using only training-history rows, then applied to test rows. HMM-based test assignments use an online filtering pass initialized from the training sequence and advanced through the embargo gap. The contrastive encoder remains a frozen offline representation in this phase, but GMM/HMM assignment models on top of those embeddings are fit fold-locally.

Phase 14A adds a compact robustness matrix. The fold-local benchmark is repeated across BTC-only, ETH-only, and BTC+ETH scopes, and across 4h, 8h, and 24h triple-barrier targets. This tests whether the main result is stable across assets and prediction horizons instead of being a single-configuration artifact.

Phase 14B adds stress robustness on top of the fold-local prediction file. It does not retrain the models; it re-scores the same out-of-sample predictions under different signal thresholds, transaction costs, and market-period slices. Bull, sideways, and bear periods are defined from rolling 30-day feature-store returns. This tests whether the benchmark conclusion survives realistic deployment assumptions.

Phase 15.0 adds artifact versioning. Latest files in `models/` are still kept for the dashboard, but frozen baselines are copied into timestamped `runs/` directories with a manifest, SHA-256 hashes, git source ref, and run-index entry. The current frozen baseline is `runs/20260522_phase14b_baseline/`.

Phase 15A adds statistical rigor. It computes fold-level IC, Sharpe, total return, drawdown, and turnover from the fold-local out-of-sample prediction file, then reports bootstrap confidence intervals and paired method tests. It also runs a DM-style Newey-West forecast-loss check on multiclass negative log-likelihood, which is a calibration-oriented complement to the IC and Sharpe tests.

Phase 15B adds multiple-testing discipline. It applies Benjamini-Hochberg and Holm corrections across tested method comparisons, labels each finding as corrected, suggestive, or not significant, and adds Probabilistic Sharpe Ratio diagnostics. This prevents a single attractive p-value from becoming an overclaimed research result.

Phase 16 adds structural regime-quality metrics independent of alpha returns. It measures regime balance, persistence, posterior confidence, pairwise NMI/ARI agreement, and agreement with the raw-feature HMM reference. The HMM sequence is a classical comparison proxy, not ground truth. This phase answers whether a method produces coherent state partitions before asking whether those states improve alpha models.

Phase 17 adds compute planning before heavier encoder experiments. It profiles a synthetic encoder forward/backward step on the local machine, estimates full 30-epoch retraining cost, and creates a capped ablation queue. On the latest CPU-only profile, one encoder retrain is estimated at about 100.10 minutes, and the full 12-run initial ablation grid is estimated at about 21.62 hours including evaluation overhead. The time-only guided runs are complete, and the Phase 22A time-frequency prototype is complete; a full time-frequency run is conditional rather than automatic.

Phase 18 adds the first encoder-objective upgrade. Instead of treating adjacent windows as positives by default, `guided_encoder.py` uses raw-feature HMM states as weak supervision: distant windows in the same HMM state become positives, and different-state windows near each other in the same symbol become harder negatives. The script writes separate guided artifacts and does not overwrite the existing `encoder.pt`, `regime_posteriors.csv`, or canonical benchmark files.

Phase 19A adds literature positioning. The project is now explicitly mapped against contrastive time-series representation learning, financial regime-switching models, financial ML validation, and regime-conditioned alpha modeling. This makes the paper contribution precise: the benchmark studies where learned regimes help, where classical HMM discipline remains stronger, and whether HMM-guided weak supervision can close that gap.

Phase 19B runs the HMM-guided encoder for the full 30-epoch budget. The guided HMM assignment reaches `HMM NMI = 0.869` and `HMM purity = 0.957`, showing that weak HMM-state supervision strongly changes the learned embedding geometry. This is structural evidence, not by itself an alpha claim.

Phase 20 is the downstream alpha retest. It uses the Phase 19B guided embeddings as frozen representations, but refits the GMM/HMM assignment layer inside each walk-forward fold before training regime-conditioned LightGBM models. The result is a strict same-universe comparison between global, raw-feature HMM, original contrastive regimes, and HMM-guided learned regimes.

Dense contrastive regime inference uses stride 1 after the encoder window warmup, so the learned-regime method is compared on the same BTC+ETH row universe as HMM-style, KMeans, and volatility-bucket baselines.

## Latest Benchmark Snapshot

Latest strict fold-local run: BTCUSDT + ETHUSDT, `tb_label_8h`, 25,920 out-of-sample rows per method.

| Method | IC | Sharpe | Drawdown | Note |
|---|---:|---:|---:|---|
| global_lgbm | 0.0024 | -0.506 | -0.688 | no-regime baseline |
| regime_lgbm_contrastive | -0.0110 | -0.834 | -0.926 | fold-local GMM on original contrastive embeddings |
| regime_lgbm_contrastive_hmm | -0.0026 | -0.548 | -0.778 | fold-local HMM on original contrastive embeddings |
| regime_lgbm_hmm | 0.0051 | -0.340 | -0.710 | raw-feature Gaussian HMM baseline |
| regime_lgbm_kmeans | 0.0072 | -0.728 | -0.860 | classical clustering baseline |
| regime_lgbm_vol_bucket | -0.0020 | -0.820 | -0.854 | volatility threshold baseline |
| regime_lgbm_hmm_guided_gmm | -0.0092 | -0.976 | -0.900 | GMM on Phase 19B guided embeddings |
| regime_lgbm_hmm_guided_hmm | 0.0094 | 0.099 | -0.614 | best point-estimate IC, Sharpe, drawdown, and return |

The current result is intentionally presented as research evidence, not a profitable trading claim. Phase 20 is the first run where the learned-regime path beats the raw-feature HMM on point-estimate alpha metrics. The improvement comes specifically from combining HMM-guided representation learning with an HMM assignment layer, not from guided embeddings plus GMM.

The statistical interpretation remains careful. `regime_lgbm_hmm_guided_hmm` has the strongest mean fold IC (`0.0080`) and the best Probabilistic Sharpe Ratio (`PSR(SR>0)=0.633`), but the fold-level IC edge over raw-feature HMM is not significant at 5% (`p=0.801`). Phase 26 preserves that honest reading: guided-HMM versus raw-feature HMM remains directionally supported, while guided-HMM versus guided-GMM becomes raw-suggestive (`IC p=0.075`) but not corrected significant.

## Fold-Local Regime Benchmark

Phase 20 extends the Phase 13 fold-local benchmark with two guided learned-regime methods. The guided embeddings are frozen from Phase 19B, while the GMM/HMM assignment layer is refit inside each fold using training-history rows only.

| Method | IC | Sharpe | Drawdown | Note |
|---|---:|---:|---:|---|
| global_lgbm | 0.0024 | -0.506 | -0.688 | no-regime baseline |
| regime_lgbm_contrastive | -0.0110 | -0.834 | -0.926 | fold-local GMM on frozen embeddings |
| regime_lgbm_contrastive_hmm | -0.0026 | -0.548 | -0.778 | fold-local HMM on frozen embeddings |
| regime_lgbm_hmm | 0.0051 | -0.340 | -0.710 | raw-feature HMM reference |
| regime_lgbm_kmeans | 0.0072 | -0.728 | -0.860 | strongest classical IC after HMM |
| regime_lgbm_vol_bucket | -0.0020 | -0.820 | -0.854 | volatility threshold baseline |
| regime_lgbm_hmm_guided_gmm | -0.0092 | -0.976 | -0.900 | guided embeddings with fold-local GMM |
| regime_lgbm_hmm_guided_hmm | 0.0094 | 0.099 | -0.614 | guided embeddings with fold-local HMM |

The Phase 20 comparison is the strongest evidence so far that the encoder upgrade matters. `hmm_guided_hmm` beats raw-feature HMM by `+0.0043` IC, `+0.439` Sharpe, `+0.095` drawdown, and `+0.567` total return in the same fold-local universe. `hmm_guided_gmm` fails, which reinforces the central finding: the useful ingredient is not just guided representation learning; it is guided representation learning plus sequential HMM state dynamics.

## Phase 14A Robustness Matrix

Phase 14A repeats the strict fold-local benchmark across 9 grid cells: 3 symbol scopes by 3 target horizons.

| Scope | Target | Best IC Method | Best IC | Best Sharpe Method | Best Sharpe | Lowest Drawdown Method |
|---|---|---|---:|---|---:|---|
| BTCUSDT | tb_label_4h | regime_lgbm_hmm_guided_hmm | 0.0058 | regime_lgbm_hmm_guided_hmm | -0.837 | regime_lgbm_hmm_guided_hmm |
| BTCUSDT | tb_label_8h | regime_lgbm_hmm_guided_gmm | 0.0004 | regime_lgbm_contrastive | -0.546 | regime_lgbm_contrastive |
| BTCUSDT | tb_label_24h | regime_lgbm_kmeans | 0.0175 | regime_lgbm_contrastive_hmm | -0.211 | regime_lgbm_kmeans |
| ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0208 | regime_lgbm_vol_bucket | 0.315 | regime_lgbm_vol_bucket |
| ETHUSDT | tb_label_8h | global_lgbm | 0.0095 | regime_lgbm_contrastive | -0.201 | regime_lgbm_hmm |
| ETHUSDT | tb_label_24h | global_lgbm | 0.0348 | global_lgbm | 0.354 | regime_lgbm_vol_bucket |
| BTCUSDT+ETHUSDT | tb_label_4h | regime_lgbm_hmm_guided_hmm | 0.0106 | regime_lgbm_hmm | -0.205 | regime_lgbm_hmm |
| BTCUSDT+ETHUSDT | tb_label_8h | regime_lgbm_hmm_guided_hmm | 0.0094 | regime_lgbm_hmm_guided_hmm | 0.099 | regime_lgbm_hmm_guided_hmm |
| BTCUSDT+ETHUSDT | tb_label_24h | regime_lgbm_contrastive_hmm | 0.0311 | regime_lgbm_vol_bucket | 0.321 | regime_lgbm_vol_bucket |

Phase 21 refreshes this matrix with the guided methods included. `regime_lgbm_hmm_guided_hmm` now wins the most IC cells (3 of 9), including the primary BTC+ETH 8-hour setting, and also wins 2 Sharpe and 2 drawdown cells. The grid is still mixed: global, KMeans, volatility buckets, and contrastive-HMM each win in some scopes. That is the right research read: guided-HMM is now a serious contender, but asset/horizon context still matters.

## Phase 14B Stress Robustness

Phase 14B reuses the strict fold-local `tb_label_8h` prediction file and stresses the trading layer across:

| Dimension | Values |
|---|---|
| Signal threshold | 0.03, 0.05, 0.07, 0.10 |
| Transaction cost | 5 bps, 10 bps, 20 bps |
| Market period | all, bull, sideways, bear |

That creates 48 stress cells and 384 method/cell rows after the Phase 21 guided-method refresh. The goal is not to find a new best backtest; it is to see whether the existing conclusion breaks when practical assumptions change.

| Metric | Most Frequent Winner | Wins |
|---|---|---:|
| Signal IC | regime_lgbm_hmm_guided_hmm | 30 |
| Sharpe | regime_lgbm_hmm_guided_hmm | 36 |
| Drawdown | regime_lgbm_hmm_guided_hmm | 28 |
| Total return | regime_lgbm_hmm_guided_hmm | 34 |

The refreshed stress test is the strongest Phase 21 evidence. Under cost, threshold, and market-period variation, guided-HMM becomes the dominant winner on all four stress metrics. This does not erase the mixed symbol/horizon matrix, but it changes the paper framing: guided-HMM is not only the best primary point estimate; it is also the most stress-robust method on the primary fold-local prediction file.

## Phase 15A/15B Statistical Tests

Phase 15A asks whether the observed method differences are statistically reliable. The test is intentionally conservative: it uses the 18 walk-forward folds as the primary unit for IC and Sharpe tests, then uses a row-level DM-style negative-log-likelihood test as a separate calibration check.

| Method | Mean Fold IC | 95% CI Low | 95% CI High | Positive IC Folds | Mean Fold Sharpe |
|---|---:|---:|---:|---:|---:|
| regime_lgbm_hmm_guided_hmm | 0.0080 | -0.0122 | 0.0278 | 9 | -0.026 |
| regime_lgbm_hmm | 0.0058 | -0.0135 | 0.0247 | 11 | -0.561 |
| regime_lgbm_kmeans | 0.0035 | -0.0202 | 0.0282 | 8 | -0.720 |
| regime_lgbm_vol_bucket | 0.0004 | -0.0230 | 0.0241 | 10 | -0.818 |
| global_lgbm | -0.0005 | -0.0207 | 0.0209 | 9 | -0.583 |
| regime_lgbm_contrastive_hmm | -0.0063 | -0.0305 | 0.0196 | 7 | -0.908 |
| regime_lgbm_hmm_guided_gmm | -0.0075 | -0.0336 | 0.0189 | 8 | -1.058 |
| regime_lgbm_contrastive | -0.0147 | -0.0373 | 0.0095 | 7 | -0.990 |

The statistical read is more cautious than the point-estimate read. `hmm_guided_hmm` has the strongest mean fold IC and the best PSR diagnostic, but its IC advantage versus raw-feature HMM is not significant at 5% (`p = 0.801`). Before correction, plain contrastive-GMM remains worse than raw-feature HMM on fold-level IC (`p = 0.035`). After Phase 15B multiple-testing correction, that negative result is suggestive rather than a hard corrected claim. This supports the current research direction without overclaiming: guided weak supervision and sequential assignment help, but broader robustness and ablation work are still needed before a paper can claim superiority.

## Phase 16 Regime Quality

Phase 16 asks whether regime methods form coherent state partitions before using them for alpha modeling. This is deliberately separate from Sharpe or return.

| Method | Balance Entropy | Switches / 1k Bars | Transition Diagonal | Avg Duration | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| contrastive | 0.999 | 32.72 | 0.967 | 30.51 | 0.032 | 0.379 |
| contrastive_hmm | 0.999 | 23.65 | 0.976 | 42.18 | 0.020 | 0.377 |
| hmm | 0.959 | 143.24 | 0.857 | 6.98 | 1.000 | 1.000 |
| kmeans | 0.866 | 246.60 | 0.753 | 4.05 | 0.182 | 0.459 |
| vol_bucket | 1.000 | 95.71 | 0.904 | 10.44 | 0.333 | 0.599 |

The useful finding is diagnostic rather than promotional. Contrastive and contrastive-HMM produce very balanced and persistent regimes, but they weakly agree with the raw-feature HMM reference. Volatility buckets and KMeans agree more with HMM states, even though they are simpler. This supports the next research step: improve the encoder objective so learned embeddings become alpha-relevant, not merely smooth.

## Phase 17 Compute Plan

Phase 17 prevents scope creep before the encoder-upgrade phases. The project now records the current encoder cost and the next ablation queue as artifacts.

| Metric | Value | Meaning |
|---|---:|---|
| Training windows | 34,798 | Sliding windows across BTCUSDT and ETHUSDT |
| Batches per epoch | 271 | Batch size 128, drop-last |
| Encoder parameters | 139,408 | Current `TemporalEncoder` size |
| Measured step time | 0.739 sec | Synthetic CPU forward/backward step |
| Estimated 30-epoch retrain | 100.10 min | One encoder experiment |
| Initial 12-run grid | 21.62 hours | 3 losses x 2 augmentations x 2 assignment methods |
| Budget status | green | Within the 24-hour local budget |

The first three runs are:

| Priority | Loss | Augmentation | Assignment | Decision |
|---:|---|---|---|---|
| 1 | hmm_guided | time_only | hmm | complete |
| 2 | hmm_guided | time_only | gmm | complete |
| 3 | hmm_guided | time_frequency | hmm | 3-epoch prototype complete |

This keeps the next step focused: the time-only guided runs are complete, and the first time-frequency guided-HMM prototype has been evaluated. The rest of the ablation grid stays on hold until the time-frequency path earns a full 30-epoch downstream retest.

## Phase 18 HMM-Guided Encoder

Phase 18 implements the first representation-learning upgrade. It uses the existing raw-feature HMM sequence as weak supervision, not ground truth. The goal is to test whether the learned embedding space becomes more state-aligned when positives and hard negatives are chosen from regime structure rather than only adjacent windows.

The one-epoch smoke run produced:

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| hmm_guided_gmm | 1 | 0.341 | 15.17 | 0.934 | 0.387 | 0.652 |
| hmm_guided_hmm | 1 | 0.353 | 18.55 | 0.946 | 0.389 | 0.620 |

This is a smoke-test result, not the final Phase 18 claim. It is still promising because the guided encoder already aligns with the HMM reference more strongly than the old contrastive encoder path from Phase 16. The full research run should use the standard 30 epochs and then feed the best guided assignments into the later statistical re-test phase.

## Phase 19A Literature Positioning

Phase 19A turns the project from a strong engineering artifact into a paper-shaped research artifact. It adds a related-work map across four areas:

- time-series contrastive learning: TS2Vec, TNC, CoST, TS-TCC, and TF-C
- financial regime switching: Hamilton-style Markov switching, Gaussian HMMs, and Markov-switching GARCH
- financial ML validation: triple-barrier labeling, purging, embargoing, and backtest overfitting control
- regime-conditioned alpha modeling: global versus state-conditioned predictors under equal validation

The key paper framing is now clear: Regime Alpha Benchmark is not claiming that HMM states are true market regimes or that the strategy is profitable. It is testing whether learned regime representations can beat, match, or explain classical sequential regimes under fair financial validation. The detailed source map is stored in `reports/related_work.md` and `reports/literature_matrix.csv`.

## Phase 19B Full Guided Encoder Run

Phase 19B runs the HMM-guided encoder for the intended 30-epoch budget. The loss fell from `0.9284` in epoch 1 to `0.0949` in epoch 30, and every batch kept valid contrastive anchors.

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| hmm_guided_gmm | 30 | 0.384 | 5.09 | 0.804 | 0.609 | 0.759 |
| hmm_guided_hmm | 30 | 0.629 | 5.72 | 0.825 | 0.869 | 0.957 |

The important structural result is that `hmm_guided_hmm` is now much closer to the raw-feature HMM reference than the old learned-regime path. Phase 16 contrastive-GMM had `HMM NMI = 0.032`; the full HMM-guided HMM reaches `0.869`. This does not yet prove better alpha performance, but it does prove that the guided objective can make the learned embedding path strongly state-aligned. The downstream test comes next.

## Phase 20 Guided Alpha Retest

Phase 20 feeds the full guided embedding space into the strict fold-local alpha benchmark. The guided encoder is not retrained inside each fold, but the regime assignment layer is: GMM and HMM state models are fit only on training-history rows and then filtered/applied into the test window.

| Method | IC | Sharpe | Drawdown | Total Return | Statistical Read |
|---|---:|---:|---:|---:|---|
| regime_lgbm_hmm | 0.0051 | -0.340 | -0.710 | -0.536 | raw-feature HMM reference |
| regime_lgbm_hmm_guided_gmm | -0.0092 | -0.976 | -0.900 | -0.854 | guided embedding + GMM fails |
| regime_lgbm_hmm_guided_hmm | 0.0094 | 0.099 | -0.614 | 0.031 | best point estimate; IC edge not significant |

The result is highly useful even though it is not a final victory lap. It says the Phase 19B structural gain can translate into better downstream alpha when the guided embedding is paired with sequential HMM filtering. It also says GMM on the same guided embedding does not work, so the finding is specifically about learned representation plus temporal state dynamics.

## Phase 22A Time-Frequency Guided Encoder

Phase 22A tests whether adding a simple frequency-domain view helps the guided encoder. Each 60-bar feature window keeps the original time-domain features and appends six low-frequency FFT magnitude bands per feature, increasing the encoder input from 22 to 154 features. This prototype uses the same HMM-guided contrastive loss, but runs only 3 epochs to check whether the idea is worth a full 30-epoch run.

| Method | Epochs | Input Features | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| tf_hmm_guided_gmm | 3 | 154 | 0.326 | 6.47 | 0.845 | 0.504 | 0.682 |
| tf_hmm_guided_hmm | 3 | 154 | 0.338 | 8.39 | 0.881 | 0.528 | 0.704 |

The result is useful but not a new winner yet. The time-frequency prototype is far stronger than the original vanilla contrastive regime path from Phase 16 (`HMM NMI = 0.032`), but it is still weaker than the full 30-epoch time-only HMM-guided encoder (`HMM NMI = 0.869`). The current conclusion is that frequency information is worth tracking as an ablation, but the project should not spend downstream alpha compute on it until a full-length time-frequency run closes the structural gap.

## Phase 23 Fold-Local Interpretability

Phase 23 adds the first paper-grade explanation layer. `src/interpretability.py` trains the same LightGBM families inside the walk-forward folds, aggregates gain/split importance, and computes capped SHAP summaries for:

- `global_lgbm`
- `regime_lgbm_hmm`
- `regime_lgbm_hmm_guided_hmm`

The important discipline is that this is fold-local interpretability. The explanation models are fit only on the training side of each fold, then aggregated across folds and regimes. That keeps the interpretation aligned with the validation setup instead of explaining a single full-sample model that has seen the future.

Top guided-HMM regime-conditioned alpha features:

| Guided-HMM Regime | Top Features | Dominant Family Read |
|---|---|---|
| 0 | `vol_20h`, `vol_of_vol`, `kurtosis`, `atr_14`, `ret_autocorr` | volatility + distribution shape |
| 1 | `vol_20h`, `vol_of_vol`, `kurtosis`, `atr_14`, `ret_autocorr` | volatility + distribution shape |
| 2 | `vol_20h`, `vol_of_vol`, `atr_14`, `kurtosis`, `ret_60h` | volatility + medium-horizon momentum |
| 3 | `vol_20h`, `vol_of_vol`, `ret_autocorr`, `skewness`, `ret_60h` | volatility + momentum/autocorrelation |

The feature-family summary is economically sensible. Guided-HMM regimes are mostly volatility-state driven, with volatility explaining about 35%-39% of SHAP share by regime, momentum about 28%-32%, and distribution shape about 11%-12%. That supports the paper narrative that the guided regimes are not arbitrary cluster IDs: the downstream alpha models mostly respond to volatility persistence, volatility-of-volatility, return autocorrelation, and return-distribution shape.

## Phase 24 Paper Protocol Freeze

Phase 24 turns the research direction into a controlled paper protocol. It does not add a new model; it freezes what the paper is allowed to ask, test, and claim before any more ablations are launched.

The protocol artifacts are:

- `reports/paper_protocol.md`: central question, dataset/feature/target/validation freeze, methods, metrics, permitted claims, forbidden claims, and decision gates.
- `reports/hypotheses.md`: the hypothesis table from H1 to H7, including which findings are supported, directional, diagnostic, or still open.
- `reports/claim_registry.md`: safe language for README/report/resume/paper text, plus claims that must not be made.
- `reports/experiment_manifest.md`: completed experiment families, future experiment queue, minimal ablation definition, and submission-readiness checklist.

The key Phase 24 decision is that the next modeling phase is now Phase 25, not Phase 24. Phase 25 should be a minimal ablation suite, and multi-asset expansion remains conditional rather than automatic.

## Phase 25 Minimal Ablation Suite

Phase 25 turns the paper mechanism into explicit ablation comparisons. It does not launch an uncontrolled retraining grid. Instead, `src/ablation_suite.py` aggregates completed structural and downstream artifacts into a compact decision table.

| Ablation Family | Question | Current Read |
|---|---|---|
| Objective guidance | Does HMM-guided weak supervision improve the learned regime path? | Supported for guided-HMM downstream alpha; mixed structurally because duration is diagnostic rather than always better |
| Assignment layer | Does HMM assignment improve learned embeddings over GMM assignment? | Strongly supported for guided embeddings, the time-frequency prototype, and downstream alpha |
| Augmentation view | Does the 3-epoch time-frequency prototype beat the full time-only guided run? | Not yet; do not expand downstream time-frequency compute yet |
| Classical reference | Does guided-HMM beat raw-feature HMM on the primary alpha benchmark? | Directionally supported; statistical refresh is still required |

The strongest Phase 25 result is that the assignment layer matters. The best learned representations become useful when they are paired with sequential HMM filtering rather than a memoryless GMM assignment. The most useful negative result is that the current time-frequency prototype is not strong enough to justify a full downstream alpha expansion.

## Phase 26 Paper Statistical Evidence Refresh

Phase 26 converts the Phase 25 ablation table into paper-facing claim tests. `src/paper_claim_tests.py` reuses the fold-level statistical metrics and tests only the comparisons needed by the paper, rather than reopening the full exploratory grid.

| Paper Claim | Phase 26 Status | Interpretation |
|---|---|---|
| HMM assignment improves the guided learned-regime alpha path | raw-suggestive | IC difference is positive and closer to significance (`p=0.075`), but not corrected significant |
| HMM assignment improves vanilla contrastive alpha | directionally supported | All focused metrics move in the right direction, but fold-level significance is weak |
| Guided-HMM beats raw-feature HMM | directionally supported | Point estimates still favor guided-HMM, but fold-level IC remains inconclusive |
| Guided-HMM beats vanilla contrastive-HMM | directionally supported | All focused metrics favor guided-HMM, but not significantly |
| Time-frequency prototype improves the guided encoder | do not expand yet | Phase 25/26 keep this as an ablation candidate, not a result claim |

The paper-safe conclusion is now sharper: the project can claim that sequential assignment is the strongest supported mechanism, and that HMM-guided learned regimes are promising versus raw-feature HMM, but it must not claim statistical dominance yet.

## Phase 27 Paper Skeleton

Phase 27 converted the evidence stack into a manuscript scaffold. After the human-reviewed Phase 37/38 synchronization, `src/paper_skeleton.py` initializes those paper artifacts only when they are missing and preserves existing manuscript, artifact-map, and checklist edits.

This phase does not add a new model or a new performance claim. It makes the paper argument auditable: each major manuscript section points back to the artifact that supports it, and the checklist keeps forbidden claims visible before the draft is moved into a venue template.

## Phase 28 Reproducibility Package

Phase 28 makes the project easier to reproduce and review. It adds `reproduce.ps1` and `reproduce.sh` with three modes:

| Mode | Command | Purpose |
|---|---|---|
| Smoke | `.\reproduce.ps1 -Mode smoke` | Compile code, verify or initialize paper artifacts, and run the validation audit |
| Full | `.\reproduce.ps1 -Mode full` | Regenerate the full local research pipeline |
| Dashboard | `.\reproduce.ps1 -Mode dashboard` | Launch the Streamlit dashboard locally |

Linux/macOS reviewers can use the equivalent shell commands: `bash reproduce.sh --mode smoke`, `bash reproduce.sh --mode full`, and `bash reproduce.sh --mode dashboard`.

It also adds `reports/environment.md`, `reports/artifact_manifest.md`, and `reports/reproduction_checklist.md`. These files separate dashboard reproduction from full research reproduction, document which artifacts are committed or ignored, and keep reviewer-facing safety checks in one place.

## Phase 29 Paper Prose Pass

Phase 29 turns `paper/main.md` from a generated scaffold into a stronger manuscript-style draft. The abstract, introduction, method narrative, validation discussion, results interpretation, robustness, interpretability, ablations, limitations, reproducibility, and conclusion now read as a coherent paper rather than a checklist.

The result language remains deliberately conservative: guided-HMM is described as the strongest point-estimate and stress-robust method, while statistical dominance over raw-feature HMM remains inconclusive.

## Phase 30 Reviewer Defense Framing

Phase 30 turns the strongest reviewer objections into explicit paper language instead of leaving them as hidden weaknesses. The project now handles three likely reviewer questions directly:

- The BTC/ETH scope is framed as a controlled crypto setting, not as a broad market-generalization claim.
- The 18 walk-forward folds are acknowledged as low-power for fold-level tests, but more defensible than treating overlapping hourly labels as independent.
- The `p=0.801` raw-feature-HMM comparison is not hidden; the headline is the mechanism that sequential assignment drives learned-regime usefulness, not a claim that guided-HMM statistically beats HMM.

This phase matters because it makes the research defensible even when results are directionally strong but not statistically decisive.

## Phase 31 Multi-Asset Universe Protocol

Phase 31 adds a pre-specified crypto expansion protocol instead of choosing extra assets after seeing results. The project now defines candidate assets, selection criteria, exclusions, and two planned universes:

- `Crypto-20`: the first controlled multi-asset generalization test.
- `Crypto-50`: the larger future expansion, gated by compute and data quality.

This protects the paper narrative from selection bias. BTC/ETH remains the controlled pilot, while Crypto-20 becomes the first broader test of whether the regime mechanism survives outside the original two assets.

The multi-asset gate is split deliberately. The Phase 20 `p=0.801` result blocks a downstream alpha generalization claim, but it does not block structural generalization diagnostics. Phase 31-35 therefore test whether the HMM-guided representation objective transfers structurally to a pre-specified wider crypto universe. A Crypto-20 fold-local alpha retest is still required before the project can claim multi-asset predictive improvement.

## Phase 32 Crypto-20 Data Pipeline And Quality Gate

Phase 32 makes the Crypto-20 universe usable in the same pipeline as the original BTC/ETH benchmark. The ingestion, feature, target, and check scripts can resolve `--universe crypto20`, then `crypto20_quality_gate.py` verifies whether every selected asset has enough bars, engineered features, labels, and acceptable gap behavior.

The current local gate passes with all 20 selected Crypto-20 assets eligible. This means the expanded universe is not just listed in a CSV; it has actually been pulled through the project’s feature and labeling machinery.

## Phase 33 Crypto-20 Classical Regime Benchmark

Phase 33 freezes the classical multi-asset baseline before retraining any learned encoder. It benchmarks raw-feature HMM, KMeans, and volatility buckets across the same Crypto-20 row universe.

The key finding is useful for the paper: KMeans gives the cleanest geometric clusters by silhouette, but its regimes are more imbalanced and less persistent. HMM has lower silhouette but stronger temporal persistence. That supports the project’s central mechanism claim: financial regimes should not be judged only by static cluster separation; sequential structure matters.

## Phase 34 Crypto-20 Guided Encoder Readiness

Phase 34 checks whether it is responsible to run the expensive Crypto-20 HMM-guided encoder experiment. It does not train the full encoder yet. Instead, it verifies that the Phase 33 HMM states provide enough weak-supervision signal for multi-asset contrastive learning.

The readiness gate passes:

- `348,606` eligible HMM-labeled encoder windows.
- All four regimes represented globally.
- `100%` positive-anchor coverage.
- `6,278,476` directed hard-negative pairs near regime boundaries.
- Estimated full 30-epoch CPU training time: about `16.77` hours.

The recommendation is to run the full Crypto-20 guided encoder next. This makes the next phase a pre-gated experiment rather than a blind compute spend.

## Phase 35 Crypto-20 Guided Encoder Training

Phase 35 is the first full learned-regime expansion beyond the BTC/ETH pilot. The run uses the Phase 33 Crypto-20 raw-feature HMM states as weak supervision, trains the HMM-guided encoder on all eligible Crypto-20 windows, and writes separate `crypto20_guided_encoder_*` artifacts so the earlier BTC/ETH guided artifacts are not overwritten.

The full CPU run completed on 348,606 eligible windows across all 20 Crypto-20 symbols. Training loss fell from `0.3258` at epoch 1 to `0.0864` at epoch 30 with valid-anchor coverage held at `1.000`.

The strongest Phase 35 assignment is `hmm_guided_hmm`:

| Method | Rows | Symbols | Silhouette | Transition diagonal | HMM NMI | HMM ARI | HMM purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hmm_guided_hmm` | 348,606 | 20 | 0.399 | 0.890 | 0.694 | 0.627 | 0.814 |
| `hmm_guided_gmm` | 348,606 | 20 | 0.230 | 0.890 | 0.506 | 0.384 | 0.721 |

This is the first multi-asset evidence that HMM-guided representation learning scales structurally beyond the BTC/ETH pilot. It supports the mechanism claim: learned embeddings become far more regime-aligned when contrastive positives and hard negatives are guided by sequential HMM state structure. It is not yet a Crypto-20 downstream alpha claim; that requires the next fold-local alpha retest.

The result is weaker-but-consistent relative to the BTC/ETH pilot. Crypto-20 `hmm_guided_hmm` has lower HMM-reference alignment than the two-asset guided-HMM run (`NMI 0.694` versus `0.869`, purity `0.814` versus `0.957`), which is expected in a more heterogeneous 20-asset universe. At the same time, the transition diagonal is higher on Crypto-20 (`0.890` versus about `0.825` in the BTC/ETH guided-HMM run), suggesting that persistence and reference-agreement can decouple at larger scale.

Full command:

```powershell
.\scripts\phases\run_phase35_crypto20_guided.ps1
```

For a short smoke/prototype run, use:

```powershell
.\scripts\phases\run_phase35_crypto20_guided.ps1 -Epochs 1 -MaxWindows 5000 -TrainOnly
```

## Phase 36 Crypto-20 Downstream Alpha Retest

Phase 36 turns the Phase 35 structural result into the next required predictive test. It feeds the Crypto-20 HMM-guided embeddings into the same fold-local, embargoed, transaction-cost-aware LightGBM alpha benchmark used in the BTC/ETH track.

The phase deliberately skips vanilla contrastive methods unless a separate dense Crypto-20 vanilla-contrastive artifact is generated. The comparison is focused on the reviewer-critical question: do HMM-guided learned regimes outperform the raw-feature HMM, KMeans, volatility buckets, and the global no-regime model on the wider pre-specified Crypto-20 universe?

The full Phase 36 run completed on 20 symbols with equal test coverage across all six methods (`230,400` OOS rows per method). The headline is mixed but useful: `hmm_guided_hmm` improves IC versus both the global model and raw-feature HMM, but it does not improve Sharpe or total return versus raw-feature HMM.

| Method | IC | Sharpe | Drawdown | Total return | OOS rows |
|---|---:|---:|---:|---:|---:|
| `global_lgbm` | 0.0175 | 0.1087 | -0.5478 | 0.0724 | 230,400 |
| `regime_lgbm_hmm` | 0.0214 | 0.1479 | -0.5428 | 0.1386 | 230,400 |
| `regime_lgbm_kmeans` | 0.0229 | 0.1238 | -0.5696 | 0.0970 | 230,400 |
| `regime_lgbm_vol_bucket` | 0.0159 | -0.1596 | -0.6539 | -0.2106 | 230,400 |
| `regime_lgbm_hmm_guided_gmm` | 0.0169 | -0.1901 | -0.6659 | -0.2412 | 230,400 |
| `regime_lgbm_hmm_guided_hmm` | 0.0226 | -0.0573 | -0.6033 | -0.1257 | 230,400 |

The paper-safe interpretation is: structural transfer from Phase 35 partially translates into predictive IC on Crypto-20, but not into superior risk-adjusted performance. This strengthens the mechanism narrative while preventing an overclaim that guided regimes dominate classical methods at portfolio level.

Full command:

```powershell
.\scripts\phases\run_phase36_crypto20_alpha.ps1
```

For a quick smoke check:

```powershell
.\scripts\phases\run_phase36_crypto20_alpha.ps1 -MaxFolds 1
```

## Phase 37 Crypto-20 Statistical Adjudication

Phase 37 asks whether the Phase 36 point-estimate differences are reliable rather than accidental. The primary unit is the walk-forward fold (`n=16`), not the `230,400` overlapping prediction rows. It adds paired fold tests, bootstrap confidence intervals, Wilcoxon and sign tests, effect sizes, Benjamini-Hochberg/Holm corrections, a time-block Newey-West DM diagnostic, Probabilistic Sharpe Ratio, and secondary per-asset heterogeneity analysis.

The result does not support a broad superiority claim. Guided-HMM has the highest mean fold IC (`0.0117`), but its edge over raw HMM is only `+0.00065` with `p=0.840`; versus global LightGBM the edge is `+0.00712` with raw `p=0.094` and corrected `q=0.322`. Its fold-level Sharpe and return differences are also inconclusive. The calibration result is negative: guided-HMM has worse multiclass NLL than global LightGBM, surviving Holm correction.

The secondary asset view is mildly encouraging but not decisive. Guided-HMM improves IC over global LightGBM in 13 of 20 assets with an average difference of `+0.00521`, but crypto assets are correlated and the sign test is not significant. The honest conclusion is structural transfer plus weak directional ranking evidence, not statistically proven multi-asset alpha dominance.

```powershell
.\scripts\phases\run_phase37_crypto20_statistics.ps1
```

## Phase 38 Research-Control Reset

Phase 38 freezes the interpretation of all results through Phase 37 and prevents the already-inspected Crypto-20 evaluation from being reused as an untouched final test. All current BTC/ETH and Crypto-20 outcomes are classified as development-observed. The next predictive phase must train scaling, weak-supervision HMMs, contrastive pairs, encoders, assignment layers, calibration, thresholds, and alpha models within the outer walk-forward boundary, with decisions made only through inner chronological validation.

The phase also restores the missing scientific ladder for the next benchmark: global and classical baselines, vanilla contrastive-GMM/HMM, and guided contrastive-GMM/HMM must use identical folds and test coverage. Calibration, soft gating, and pooled experts are authorized only after this fully fold-local baseline passes. Crypto-50 expansion, unrestricted tuning, and product deployment remain deferred.

Control artifacts:

- `reports/phase38_master_protocol.md`
- `reports/data_role_registry.csv`
- `reports/experiment_ledger.csv`
- `reports/publication_acceptance_gates.md`
- `reports/phase39_fold_local_encoder_design.md`

## Phase 39R Repaired Fully Fold-Local Encoder

Phase 39 originally implemented fold-local fitting for the feature scaler, weak-supervision HMM, contrastive pair construction, vanilla and guided encoders, sequential/non-sequential assignment layers, and downstream LightGBM models. Phase 39R is the repaired version of that milestone.

The repair matters because the first full Phase 39 Crypto-20 run used per-symbol positional folds. Each symbol looked separated individually, but when all 20 assets were pooled, the training and test sets overlapped heavily in real calendar time. That made the original result unsafe as scientific evidence.

The original full artifact covers 16 folds and equal method coverage but is invalidated as predictive evidence because per-symbol positional folds overlapped in calendar time. It is retained only as audit history. The repaired common-calendar implementation, data freeze, unit suite, classical baseline, and full 16-fold neural/guided run are now complete under `crypto20-development-v1`.

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -p test_*.py -v
.\scripts\phases\run_phase39_fold_local_encoder.ps1 -Epochs 1 -BatchSize 128 -MaxWindows 5000 -MaxFolds 1 -RunName phase39_resume_smoke
.\scripts\phases\run_phase39_fold_local_encoder.ps1 -Epochs 1 -BatchSize 128 -MaxWindows 5000 -MaxFolds 1 -RunName phase39_resume_smoke -Resume
```

The repaired development protocol uses 16 folds, up to 30 epochs with inner-validation early stopping, batch size 128, seed 42, and a pre-frozen 5,000-window encoder budget. Training is additionally bound to `crypto20-development-v1`; it stops on changed code, data, configuration, asset order, folds, freeze hash, or checkpoint hash. The repaired full run found weak/inconclusive downstream alpha rather than a positive trading result.

Phase 39R completed artifacts include:

- repaired development freeze manifest and fold calendar;
- repaired classical baseline results;
- repaired 16-fold neural/guided fold-local results;
- repaired method comparison and coverage files;
- automated research-grade check reports;
- claim-control documentation that prevents the invalidated run from being used as evidence.

## Phase 40 Repaired Statistical Adjudication

Phase 40 statistically adjudicates the repaired Phase 39R outputs. Its job is not to rescue a positive result; its job is to decide what the repaired evidence can honestly support.

The result is deliberately conservative:

- repaired development results remain weak/negative;
- no corrected IC/Sharpe dominance claim is supported;
- bootstrap intervals and paired fold tests show high uncertainty;
- development-observed evidence cannot be treated as an untouched final test.

Core artifacts:

- `reports/phase40_repaired_statistical_adjudication.md`
- `models/crypto20_repaired_fold_local_statistical_method_summary.csv`
- `models/crypto20_repaired_fold_local_statistical_claims.csv`
- `models/crypto20_repaired_fold_local_statistical_multiple_testing.csv`

## Phase 41 And 41B Bounded Improvement Protocol

Phase 41 registers a bounded improvement path after the weak repaired result. The key rule is that candidate selection must happen inside each outer fold using inner chronological validation only. It does not authorize tuning on repaired outer-test results.

Phase 41B runs the global/classical probability-calibration and soft-gating subset. Score-threshold execution control is registered but deferred and excluded from the final locked candidate.

The outcome remains negative/diagnostic:

- candidate selection is validation-safe;
- the run does not improve the repaired alpha conclusion;
- no corrected IC/Sharpe dominance claim is supported;
- Phase 41B does not become the final locked method.

Core artifacts:

- `reports/phase41_bounded_improvement_protocol.md`
- `reports/phase41_inner_validation_candidate_run.md`
- `models/phase41_candidate_registry.csv`
- `models/phase41_classical_experiment_results.csv`
- `models/phase41_classical_statistical_claims.csv`

## Phase 42 Interpretation And Execution Hardening

Phase 42 explains why the repaired alpha result is weak. It is a diagnostic phase, not a new model-search phase.

It checks:

- transaction-cost and score-threshold sensitivity;
- regime transition behavior;
- cross-asset alpha fragility;
- feature-family target alignment.

The main conclusion is that the weak alpha result is not explained away by one simple calibration or execution bug. Execution assumptions, regime transitions, and cross-asset heterogeneity all matter. Phase 42 does not make a tradability claim.

Core artifacts:

- `reports/phase42_interpretation_execution_hardening.md`
- `models/phase42_execution_stress_summary.csv`
- `models/phase42_regime_transition_diagnostics.csv`
- `models/phase42_cross_asset_alpha_diagnostics.csv`
- `models/phase42_feature_family_diagnostics.csv`

## Phase 43A Final Candidate And Locked-Holdout Freeze

Phase 43A freezes the final candidate before any locked-holdout outcome is inspected.

Frozen final candidate:

```text
regime_lgbm_hmm_guided_hmm
```

Primary references:

```text
global_lgbm
regime_lgbm_hmm
```

Excluded after the freeze:

- calibration rescue;
- soft-gating rescue;
- score-threshold rescue;
- new feature selection;
- new label or horizon selection;
- new architecture search;
- switching to another method after locked results are known.

Core artifacts:

- `configs/phase43_locked_holdout_freeze_v1.json`
- `reports/phase43_locked_holdout_freeze.md`
- `models/phase43_locked_candidate_manifest.csv`
- `models/phase43_locked_claim_rules.csv`
- `models/phase43_locked_holdout_rules.csv`

## Phase 43B Locked External Holdout Evaluation

Phase 43B registers, freezes, evaluates, and adjudicates a 10-asset external locked holdout. The external assets are selected through prewritten coverage and quality rules before model outcomes are inspected.

The one-shot locked evaluation is complete:

- 10 registered external assets;
- 18 folds;
- 129,600 out-of-sample rows per method;
- frozen candidate compared against `global_lgbm` and `regime_lgbm_hmm`;
- prewritten relative IC/Sharpe rule is satisfied;
- positive tradable alpha is not supported because locked Sharpe and total return remain negative.

The allowed paper claim is narrow: the frozen guided-HMM candidate receives limited locked relative support. The paper must not claim a profitable/deployable strategy, broad dominance, or permission to retune on the same locked holdout.

Core artifacts:

- `reports/phase43b_locked_holdout_registration.md`
- `reports/phase43b_locked_holdout_data_freeze.md`
- `reports/phase43b_locked_external_evaluation.md`
- `reports/phase43b_locked_external_adjudication.md`
- `models/phase43b_locked_external_experiment_results.csv`
- `models/phase43b_locked_external_primary_comparison.csv`
- `models/phase43b_locked_external_claims.csv`

## Phase 44 Paper-Readiness Evidence Package

Phase 44 turns the repaired development evidence and locked external holdout into a paper-facing package. It does not tune models, change labels, change candidate choice, or touch final/locked evaluation data.

Phase 44 adds:

- a manuscript draft synchronized to the repaired and locked evidence;
- a paper evidence matrix separating development-observed and locked evidence;
- a submission risk register;
- a reviewer brief with safe answers to likely objections;
- a pre-push hardening audit;
- automated research-grade checks for Phase 44 claim-control wording.

Current paper story:

> HMM-guided contrastive regimes receive limited locked-holdout relative support, but the evidence does not support a profitable or deployable trading strategy. The main contribution is the validation repair, benchmark discipline, locked-holdout adjudication, and mechanism boundary.

Core artifacts:

- `paper/main.md`
- `reports/phase44_paper_readiness_package.md`
- `reports/phase44_reviewer_brief.md`
- `reports/phase44_prepush_hardening_audit.md`
- `models/phase44_paper_evidence_matrix.csv`
- `models/phase44_submission_risk_register.csv`

## Phase 45 Venue-Ready Manuscript Package

Phase 45 is paper packaging, not model rescue. It turns the Phase 44 evidence story into a venue-facing manuscript package while preserving the locked-holdout claim boundaries.

Phase 45 adds:

- a venue-ready structured manuscript source;
- a table plan for the final paper;
- a figure plan for the final paper;
- a claim-to-section map that blocks profitability, dominance, and candidate-switching overclaims;
- a venue requirement audit for ICAIF/ACM-style page, anonymity, template, self-contained submission, and artifact-availability constraints;
- a reproducibility appendix;
- a submission checklist for final formatting and reviewer safety.

The paper claim remains intentionally narrow:

> HMM-guided contrastive regimes receive limited locked relative support under the prewritten Phase 43B rule, but the evidence does not support a profitable or deployable trading strategy.

Core artifacts:

- `paper/phase45_venue_ready_manuscript.md`
- `reports/phase45_venue_manuscript_package.md`
- `reports/phase45_reproducibility_appendix.md`
- `reports/phase45_submission_checklist.md`
- `reports/phase45_external_research_audit.md`
- `models/phase45_table_plan.csv`
- `models/phase45_figure_plan.csv`
- `models/phase45_claim_to_section_map.csv`
- `models/phase45_venue_requirement_audit.csv`

## Phase 46 Final Research Completion Package

Phase 46 is the final research-completion layer. It is not a new model experiment and does not rerun, tune, or reinterpret locked/final evaluation data.

Phase 46 adds:

- a compact eight-page section budget for the final paper;
- an anonymous ACM `sigconf` LaTeX skeleton;
- a final submission manuscript source;
- a submission gate matrix;
- an anonymity audit;
- a final claim audit;
- a reviewer objection/response pack;
- a final submission TODO list.

The Phase 46 status is deliberately precise:

> The repository evidence package is research-complete and ready for final paper formatting. It is not ready for blind external submission until the final PDF is compiled, page length is measured, citations and figures are finalized, an anonymity audit is run on the PDF/metadata, and an artifact DOI/archive decision is made.

Core artifacts:

- `paper/phase46_submission_manuscript.md`
- `paper/phase46_acm_sigconf_skeleton.tex`
- `reports/phase46_final_research_completion.md`
- `reports/phase46_submission_readiness_audit.md`
- `reports/phase46_anonymity_audit.md`
- `reports/phase46_reviewer_response_pack.md`
- `reports/phase46_final_submission_todo.md`
- `models/phase46_section_budget.csv`
- `models/phase46_submission_gate_matrix.csv`
- `models/phase46_anonymity_audit.csv`
- `models/phase46_final_claim_audit.csv`
- `models/phase46_reviewer_objection_matrix.csv`

## Phase 47 Submission Manuscript Build

Phase 47 is a submission-build and blind-review-hardening layer. It is not a new experiment and does not tune models, change labels, select a new candidate, rerun the locked holdout, or reinterpret locked/final evaluation data.

Phase 47 adds:

- a build-oriented anonymous ACM `sigconf` LaTeX draft;
- a BibTeX reference file with starter citations;
- a manuscript build audit;
- a table manifest mapping final tables to evidence artifacts;
- a figure manifest identifying which figures still need final drawing;
- a source-level anonymity audit;
- a locally compiled PDF evidence artifact;
- a camera-ready/submission gap list.

The Phase 47 status is intentionally precise:

> The manuscript source is now build-oriented and paper-safe, and the local ACM draft PDF compiles to a 2-page artifact. It is not yet a submitted paper: current venue-template verification, LaTeX/BibTeX warning cleanup, PDF/metadata anonymity review, final figure drawing, citation polish, and artifact archive/DOI decisions remain before external submission.

Core artifacts:

- `paper/phase47_submission_draft.tex`
- `paper/phase47_references.bib`
- `paper/phase47_submission_draft.pdf`
- `reports/phase47_submission_build_report.md`
- `reports/phase47_blind_review_hardening.md`
- `reports/phase47_camera_ready_gap_list.md`
- `models/phase47_manuscript_build_audit.csv`
- `models/phase47_table_manifest.csv`
- `models/phase47_figure_manifest.csv`
- `models/phase47_anonymity_source_audit.csv`
- `models/phase47_reference_manifest.csv`

## Phase 48 Review-Ready Paper Polish

Phase 48 turns the compiled Phase 47 draft into a cleaner review-ready manuscript package. It is not a model-rescue phase and does not tune models, change labels, select a new candidate, rerun the locked holdout, or reinterpret locked/final evaluation data.

Phase 48 adds:

- a polished anonymous ACM manuscript source;
- a dedicated Phase 48 BibTeX file;
- a locally compiled review PDF;
- a claim-traceability audit that allows only limited locked relative support;
- a paper-quality audit for anonymity, locked-holdout boundaries, candidate-switching boundaries, and negative-result framing;
- a LaTeX/PDF warning audit and warning-resolution checklist;
- a reviewer reading guide for allowed and blocked claims.

The Phase 48 status is intentionally conservative:

> The review-ready manuscript package exists and compiles locally, but it is not a submitted paper. The allowed scientific claim remains limited locked relative IC/Sharpe support plus validation-repair discipline. Profitable-alpha, deployment, candidate-switching, and same-holdout rescue claims remain blocked.

Core artifacts:

- `paper/phase48_review_ready_manuscript.tex`
- `paper/phase48_references.bib`
- `paper/phase48_review_ready_manuscript.pdf`
- `reports/phase48_paper_polish_report.md`
- `reports/phase48_reviewer_reading_guide.md`
- `reports/phase48_latex_warning_resolution.md`
- `models/phase48_paper_quality_audit.csv`
- `models/phase48_claim_traceability.csv`
- `models/phase48_latex_warning_audit.csv`
- `models/phase48_section_budget.csv`

## Phase 49 Final Submission Closure

Phase 49 closes the specific remaining submission gaps identified after Phase 48. It is not a new experiment and does not tune models, change labels, select a new candidate, rerun the locked holdout, or reinterpret final evaluation data.

Phase 49 adds:

- a final F1 validation-repair figure showing invalidated positional folds versus repaired common-calendar folds;
- a final F2 system-overview figure showing features, fold-local HMM weak labels, contrastive windows, HMM-guided encoder, regime assignment, LightGBM, and claim-audited metrics;
- expanded time-series contrastive-learning BibTeX coverage for TS2Vec, TNC, CoST, TS-TCC, and TF-C;
- a submission-ready anonymous ACM manuscript source and locally compiled PDF;
- a PDF metadata anonymity audit;
- a LaTeX/PDF warning-state audit;
- an artifact-release guide that keeps the artifact DOI as a human step before any availability claim; the artifact DOI is still a human step.

The Phase 49 status is intentionally honest:

> The repository-side submission package now contains the final conceptual figures, citation expansion, compiled anonymous PDF target, metadata audit, and artifact-release checklist. It is still not an externally submitted paper until a human/advisor performs the final visual layout pass, verifies the active venue template, and creates or withholds a DOI archive according to the venue’s double-blind rules.

Core artifacts:

- `paper/phase49_submission_ready_manuscript.tex`
- `paper/phase49_references.bib`
- `paper/phase49_submission_ready_manuscript.pdf`
- `paper/figures/phase49_f1_validation_repair.svg`
- `paper/figures/phase49_f2_system_overview.svg`
- `reports/phase49_final_submission_closure.md`
- `reports/phase49_pdf_metadata_audit.md`
- `reports/phase49_artifact_release_guide.md`
- `models/phase49_submission_closure_audit.csv`
- `models/phase49_figure_manifest.csv`
- `models/phase49_citation_audit.csv`
- `models/phase49_pdf_metadata_audit.csv`
- `models/phase49_latex_warning_audit.csv`
- `models/phase49_artifact_release_checklist.csv`

## Current Status

The original Phase 39 result table is retained for audit history but is not scientific evidence because its per-symbol positional folds overlapped in calendar time. The repaired calendar-aligned classical and neural/guided benchmarks are complete, all methods have equal coverage, and the repaired Phase 40 statistical adjudication is complete. Phase 41/41B has registered bounded candidates and run the global/classical calibration plus soft-gating subset; score-threshold execution control remains deferred and is excluded from the final locked candidate. The result remains weak/negative; corrected IC/Sharpe dominance is unsupported. Phase 42 explains the weak result through execution sensitivity, regime-transition behavior, cross-asset fragility, and feature-family target alignment without making a tradability claim. Phase 43A freezes `regime_lgbm_hmm_guided_hmm` as the single final guided-HMM mechanism candidate before any locked-holdout outcome is inspected. Phase 43B registers and freezes a 10-asset external locked holdout, runs the one-shot 18-fold locked evaluation, and adjudicates the result: the frozen guided-HMM candidate satisfies the prewritten relative IC/Sharpe rule versus the two primary references, but negative Sharpe and total return mean no tradable-positive strategy claim is supported. Phase 44 converts this into a paper-ready evidence package, risk register, and updated manuscript draft. Phase 45 converts that story into a venue-ready manuscript package, table/figure plan, venue requirement audit, external research audit, reproducibility appendix, and submission checklist while preserving the same claim boundaries. Phase 46 marks the repository evidence package research-complete and creates the final paper-formatting, anonymity, reviewer-response, and submission-gate package. Phase 47 converts that package into a build-oriented anonymous LaTeX draft, references file, compiled PDF, source anonymity audit, table/figure manifests, and submission gap list. Phase 48 polishes that into a review-ready manuscript package with claim traceability, PDF warning audit, reviewer reading guide, and an explicit no-rescue/no-deployment boundary. Phase 49 closes the remaining repository-side submission gaps by adding final F1/F2 figures, citation coverage, compiled submission-ready source/PDF, PDF metadata audit, LaTeX warning-state audit, and an artifact-release guide. It remains not externally submitted until the current venue template is verified by a human/advisor, the PDF receives a final visual pass, and the artifact DOI decision is completed under the venue’s double-blind rules. The research-grade regression gate passes.

Use the frozen-universe calendar audit command, not the bare default command:

```powershell
.\env\Scripts\python.exe src\fold_local_encoder_walkforward.py --universe crypto20 --calendar-audit-only
```
