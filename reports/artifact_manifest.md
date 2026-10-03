# Regime Alpha Benchmark Artifact Manifest

## Purpose

This manifest explains which artifacts are committed, which are regenerated locally, and which are intentionally excluded from GitHub.

## Committed Curated Artifacts

Committed artifacts are small summary files or visual outputs used by the dashboard, README, report, and paper scaffold.

| Category | Examples | Role |
|---|---|---|
| Target diagnostics | `models/target_distribution.csv`, `models/target_quality.csv` | Label quality and class balance |
| Regime diagnostics | `models/regime_benchmark_summary.csv`, `models/regime_quality_summary.csv` | Structural regime comparison |
| Fold-local alpha results | `models/walkforward_experiment_results.csv`, `models/guided_alpha_comparison.csv` | Main predictive benchmark |
| Robustness | `models/robustness_summary.csv`, `models/robustness_stress_summary.csv` | Horizon/scope and stress checks |
| Statistical evidence | `models/statistical_test_summary.csv`, `models/paper_statistical_summary.csv` | Paper-safe claim tests |
| Interpretability | `models/feature_importance_by_regime.csv`, `models/feature_family_summary.csv` | Fold-local feature attribution |
| Phase 38 control layer | `reports/phase38_master_protocol.md`, `reports/data_role_registry.csv`, `reports/experiment_ledger.csv`, `reports/publication_acceptance_gates.md` | Data-role, experiment-lineage, scope, and acceptance control |
| Phase 39 implementation contract | `reports/phase39_fold_local_encoder_design.md` | Exact fold-local encoder boundaries, modules, artifacts, tests, and compute gates |
| Phase 39 invalidated historical evidence | `models/crypto20_fold_local_encoder_manifest.csv`, `models/crypto20_fold_local_encoder_coverage.csv`, `models/crypto20_fold_local_encoder_loss.csv`, `models/crypto20_fold_local_fold_metrics.csv`, `models/crypto20_fold_local_experiment_results.csv`, `models/crypto20_fold_local_statistical_method_summary.csv`, `models/crypto20_fold_local_statistical_claims.csv`, `reports/phase39_fold_local_results.md` | Computationally complete but invalidated by cross-asset calendar overlap; retained only for audit history |
| Repaired development-data freeze | `configs/crypto20_development_freeze_v1.json`, `models/crypto20_development_freeze_manifest.json`, `models/crypto20_development_symbol_manifest.csv`, `models/crypto20_development_fold_calendar.csv`, `models/crypto20_development_universe_frozen.csv`, `reports/crypto20_development_data_freeze.md` | Immutable development-only asset, timestamp, row, database, experiment-data, and fold lineage for repaired baselines |
| Repaired classical gate | `src/repaired_classical_baseline.py`, `scripts/phases/run_phase39r_classical_baseline.ps1`, `scripts/phases/run_phase39r_classical_baseline.sh`, `models/crypto20_repaired_classical_experiment_results.csv`, `models/crypto20_repaired_classical_fold_metrics.csv`, `models/crypto20_repaired_classical_coverage.csv`, `models/crypto20_repaired_classical_manifest.csv`, `models/crypto20_repaired_classical_implementations.csv`, `reports/phase39r_classical_baseline_protocol.md` | Completed frozen global/HMM/KMeans/volatility benchmark with per-fold checkpoints before neural retraining; classical baselines are weak/negative under repaired IC and portfolio diagnostics |
| Repaired neural/guided gate | `models/crypto20_repaired_fold_local_experiment_results.csv`, `models/crypto20_repaired_fold_local_fold_metrics.csv`, `models/crypto20_repaired_fold_local_encoder_manifest.csv`, `models/crypto20_repaired_fold_local_encoder_coverage.csv`, `models/crypto20_repaired_fold_local_guided_comparison.csv`, `reports/phase39r_neural_fold_local_results.md` | Completed 16-fold fold-local neural/guided development benchmark under the repaired freeze; alpha remains weak/inconclusive and is not a final-test claim |
| Phase 40 repaired statistical adjudication | `models/crypto20_repaired_fold_local_statistical_method_summary.csv`, `models/crypto20_repaired_fold_local_statistical_pairwise_tests.csv`, `models/crypto20_repaired_fold_local_statistical_multiple_testing.csv`, `models/crypto20_repaired_fold_local_statistical_claims.csv`, `models/crypto20_repaired_fold_local_statistical_sharpe_diagnostics.csv`, `reports/phase40_repaired_statistical_adjudication.md` | Corrected fold/asset/NLL/PSR adjudication of the repaired Phase 39R outputs; no robust positive-alpha or method-dominance claim is supported |
| Phase 41 bounded improvement protocol | `configs/phase41_bounded_candidates_v1.json`, `src/phase41_bounded_candidates.py`, `models/phase41_candidate_registry.csv`, `models/phase41_selection_rules.csv`, `reports/phase41_bounded_improvement_protocol.md`, `tests/test_phase41_bounded_candidates.py` | Registers probability-calibration, soft-gating, and deferred score-threshold execution-control candidates with inner-validation-only selection rules; no performance claim and no Phase 40 outer-test tuning |
| Phase 41B inner-validation runner | `src/phase41_inner_validation_candidates.py`, `scripts/phases/run_phase41_inner_validation_candidates.ps1`, `scripts/phases/run_phase41_inner_validation_candidates.sh`, `tests/test_phase41_inner_validation_candidates.py` | Implements inner-validation-selected probability calibration and soft-gating candidates for the global/classical ladder; score-threshold execution control is explicitly deferred; one-fold smoke is diagnostic only and full 16-fold run is explicit |
| Phase 41B full candidate result | `models/phase41_classical_experiment_results.csv`, `models/phase41_classical_fold_metrics.csv`, `models/phase41_classical_selected_candidates.csv`, `models/phase41_classical_inner_candidate_results.csv`, `models/phase41_classical_statistical_method_summary.csv`, `models/phase41_classical_statistical_claims.csv`, `reports/phase41_inner_validation_candidate_run.md` | Completed 16-fold global/classical candidate run selected by inner validation only; no corrected IC/Sharpe dominance or positive-alpha claim is supported |
| Phase 42 interpretation and execution hardening | `src/phase42_interpretation_execution.py`, `scripts/phases/run_phase42_interpretation_execution.ps1`, `models/phase42_execution_stress_summary.csv`, `models/phase42_regime_transition_diagnostics.csv`, `models/phase42_stable_transition_alpha.csv`, `models/phase42_cross_asset_alpha_diagnostics.csv`, `models/phase42_feature_family_diagnostics.csv`, `reports/phase42_interpretation_execution_hardening.md` | Development-observed diagnostics for execution sensitivity, regime transitions, cross-asset fragility, and feature-family target alignment; explains weak alpha without claiming tradability |
| Phase 43A locked holdout freeze | `configs/phase43_locked_holdout_freeze_v1.json`, `src/phase43_locked_holdout_freeze.py`, `models/phase43_locked_candidate_manifest.csv`, `models/phase43_locked_claim_rules.csv`, `models/phase43_locked_holdout_rules.csv`, `reports/phase43_locked_holdout_freeze.md` | Freezes `regime_lgbm_hmm_guided_hmm` as the one final guided-HMM mechanism candidate before any locked holdout outcome is inspected; excludes calibration, soft-gating, and threshold rescue tuning |
| Phase 43B locked holdout registration | `configs/phase43b_locked_holdout_registration_v1.json`, `src/phase43b_locked_holdout_registration.py`, `models/phase43b_holdout_candidate_quality.csv`, `models/phase43b_registered_holdout_symbols.csv`, `models/phase43b_locked_holdout_registration_manifest.csv`, `reports/phase43b_locked_holdout_registration.md` | Registers and hashes locked-holdout membership readiness before model evaluation; status is `registered_ready` with 10 external assets |
| Phase 43B locked holdout freeze and evaluation | `src/phase43b_locked_holdout_freeze.py`, `models/phase43b_locked_holdout_freeze_manifest.json`, `models/phase43b_locked_holdout_symbol_manifest.csv`, `models/phase43b_locked_external_experiment_results.csv`, `models/phase43b_locked_external_fold_metrics.csv`, `reports/phase43b_locked_external_evaluation.md` | Freezes the registered external holdout and executes the full 18-fold locked evaluation without same-holdout tuning |
| Phase 43B locked adjudication | `src/phase43b_locked_holdout_adjudication.py`, `models/phase43b_locked_external_primary_comparison.csv`, `models/phase43b_locked_external_claims.csv`, `reports/phase43b_locked_external_adjudication.md` | Applies the Phase 43A frozen success/failure rule: limited relative IC/Sharpe support is satisfied, but no tradable-positive strategy claim is supported |
| Phase 44 paper-readiness package | `src/phase44_paper_readiness_package.py`, `models/phase44_paper_evidence_matrix.csv`, `models/phase44_submission_risk_register.csv`, `reports/phase44_paper_readiness_package.md`, `reports/phase44_reviewer_brief.md`, `reports/phase44_prepush_hardening_audit.md`, `paper/main.md`, `reports/paper_submission_checklist.md` | Converts repaired-development and locked-holdout evidence into a paper-facing package with explicit claim boundaries, reviewer-response guidance, risk controls, pre-push hardening, and no model-rescue tuning |
| Phase 45 venue-ready manuscript package | `src/phase45_venue_manuscript_package.py`, `paper/phase45_venue_ready_manuscript.md`, `reports/phase45_venue_manuscript_package.md`, `reports/phase45_external_research_audit.md`, `reports/phase45_reproducibility_appendix.md`, `reports/phase45_submission_checklist.md`, `models/phase45_table_plan.csv`, `models/phase45_figure_plan.csv`, `models/phase45_claim_to_section_map.csv`, `models/phase45_venue_requirement_audit.csv` | Converts the Phase 44 evidence story into a venue-facing manuscript structure, table/figure plan, ICAIF/ACM-style venue requirement audit, reproducibility appendix, and submission checklist without model rescue, candidate switching, or same-holdout retuning |
| Phase 46 final research completion package | `src/phase46_final_research_completion.py`, `scripts/phases/run_phase46_final_research_completion.ps1`, `scripts/phases/run_phase46_final_research_completion.sh`, `paper/phase46_submission_manuscript.md`, `paper/phase46_acm_sigconf_skeleton.tex`, `reports/phase46_final_research_completion.md`, `reports/phase46_submission_readiness_audit.md`, `reports/phase46_anonymity_audit.md`, `reports/phase46_reviewer_response_pack.md`, `reports/phase46_final_submission_todo.md`, `models/phase46_section_budget.csv`, `models/phase46_submission_gate_matrix.csv`, `models/phase46_anonymity_audit.csv`, `models/phase46_final_claim_audit.csv`, `models/phase46_reviewer_objection_matrix.csv` | Marks the repository evidence package research-complete and creates the final paper-formatting, anonymity, reviewer-response, claim-audit, and submission-gate package without model rescue or final-data modification |
| Phase 47 submission manuscript build | `src/phase47_submission_build.py`, `scripts/phases/run_phase47_submission_build.ps1`, `scripts/phases/run_phase47_submission_build.sh`, `paper/phase47_submission_draft.tex`, `paper/phase47_references.bib`, `paper/phase47_submission_draft.pdf`, `reports/phase47_submission_build_report.md`, `reports/phase47_blind_review_hardening.md`, `reports/phase47_camera_ready_gap_list.md`, `models/phase47_manuscript_build_audit.csv`, `models/phase47_table_manifest.csv`, `models/phase47_figure_manifest.csv`, `models/phase47_anonymity_source_audit.csv`, `models/phase47_reference_manifest.csv` | Converts the Phase 46 completion package into a build-oriented anonymous manuscript source, compiled local PDF evidence, and submission audit without model rescue, candidate switching, or locked-data reinterpretation |
| Phase 48 review-ready paper polish | `src/phase48_paper_polish.py`, `scripts/phases/run_phase48_paper_polish.ps1`, `scripts/phases/run_phase48_paper_polish.sh`, `paper/phase48_review_ready_manuscript.tex`, `paper/phase48_references.bib`, `paper/phase48_review_ready_manuscript.pdf`, `reports/phase48_paper_polish_report.md`, `reports/phase48_reviewer_reading_guide.md`, `reports/phase48_latex_warning_resolution.md`, `models/phase48_paper_quality_audit.csv`, `models/phase48_claim_traceability.csv`, `models/phase48_latex_warning_audit.csv`, `models/phase48_section_budget.csv` | Polishes the compiled manuscript into a review-ready package with claim traceability, PDF warning audit, section budget, and reviewer reading guide while preserving the no-rescue/no-deployment boundary |
| Phase 49 final submission closure | `src/phase49_submission_closure.py`, `scripts/phases/run_phase49_submission_closure.ps1`, `scripts/phases/run_phase49_submission_closure.sh`, `paper/phase49_submission_ready_manuscript.tex`, `paper/phase49_references.bib`, `paper/phase49_submission_ready_manuscript.pdf`, `paper/figures/phase49_f1_validation_repair.svg`, `paper/figures/phase49_f2_system_overview.svg`, `reports/phase49_final_submission_closure.md`, `reports/phase49_pdf_metadata_audit.md`, `reports/phase49_artifact_release_guide.md`, `models/phase49_submission_closure_audit.csv`, `models/phase49_figure_manifest.csv`, `models/phase49_citation_audit.csv`, `models/phase49_pdf_metadata_audit.csv`, `models/phase49_latex_warning_audit.csv`, `models/phase49_artifact_release_checklist.csv` | Closes the remaining repository-side submission gaps by adding final F1/F2 figures, time-series citation coverage, compiled anonymous PDF target, PDF metadata audit, LaTeX warning-state audit, and artifact-release DOI boundary without changing models or final data |
| Research-grade regression gate | `src/research_grade_checks.py`, `run_research_grade_checks.ps1`, `run_research_grade_checks.sh`, `models/research_grade_check_report.csv`, `models/research_grade_check_report.md` | Fast artifact gate plus full freeze/unit/calendar gate to detect whether future features break existing research guarantees |
| Multi-asset protocol | `configs/crypto_universe_candidates.csv`, `models/asset_universe_crypto20.csv`, `models/crypto20_data_quality.csv`, `reports/crypto20_pipeline_plan.md` | Pre-specified crypto-universe expansion and data-quality gate |
| Crypto-20 regimes | `models/crypto20_regime_benchmark_summary.csv`, `models/crypto20_regime_symbol_summary.csv` | First multi-asset regime benchmark before learned-regime retraining |
| Crypto-20 guided readiness | `models/crypto20_guided_pair_summary.csv`, `models/crypto20_guided_compute_plan.csv`, `models/crypto20_guided_gate.csv` | Pair-mining and compute gate before full Crypto-20 guided encoder training |
| Crypto-20 guided encoder | `models/crypto20_guided_encoder_summary.csv`, `models/crypto20_guided_encoder_loss.csv`, `models/crypto20_guided_encoder_comparison.csv`, `models/crypto20_guided_encoder_loss_curve.png`, `models/crypto20_guided_encoder_transition_hmm_guided_gmm.png`, `models/crypto20_guided_encoder_transition_hmm_guided_hmm.png` | Phase 35 full Crypto-20 HMM-guided encoder structural result |
| Crypto-20 alpha retest | `models/crypto20_walkforward_experiment_results.csv`, `models/crypto20_walkforward_regime_summary.csv`, `models/crypto20_walkforward_guided_alpha_comparison.csv`, `models/crypto20_walkforward_equity_curve.png`, `reports/crypto20_alpha_generalization.md` | Phase 36 fold-local downstream alpha retest |
| Crypto-20 statistical evidence | `models/crypto20_statistical_method_summary.csv`, `models/crypto20_statistical_claims.csv`, `models/crypto20_statistical_asset_metrics.csv`, `reports/crypto20_statistical_protocol.md` | Phase 37 fold uncertainty, corrected tests, calibration, and asset heterogeneity |
| Paper package | `paper/main.md`, `reports/paper_artifact_map.csv`, `reports/paper_submission_checklist.md` | Manuscript scaffold and submission planning |
| Reproduction helpers | `reproduce.ps1`, `reproduce.sh`, `scripts/phases/run_phase35_crypto20_guided.ps1`, `scripts/phases/run_phase35_crypto20_guided.sh`, `scripts/phases/run_phase36_crypto20_alpha.ps1`, `scripts/phases/run_phase36_crypto20_alpha.sh`, `scripts/phases/run_phase37_crypto20_statistics.ps1`, `scripts/phases/run_phase37_crypto20_statistics.sh`, `scripts/phases/run_phase41_bounded_candidates.ps1`, `scripts/phases/run_phase41_inner_validation_candidates.ps1`, `scripts/phases/run_phase43b_locked_holdout_registration.ps1`, `scripts/phases/run_phase43b_locked_holdout_freeze.ps1`, `scripts/phases/run_phase43b_locked_holdout_adjudication.ps1`, `scripts/phases/run_phase44_paper_readiness_package.ps1`, `scripts/phases/run_phase45_venue_manuscript_package.ps1`, `scripts/phases/run_phase46_final_research_completion.ps1`, `scripts/phases/run_phase47_submission_build.ps1`, `scripts/phases/run_phase48_paper_polish.ps1`, `scripts/phases/run_phase49_submission_closure.ps1` | Cross-platform smoke/full/dashboard and long-run phase helpers |
| Versioning | `runs/run_index.csv`, `runs/*/manifest.json`, `runs/*/artifact_manifest.csv` | Frozen curated snapshots |

## Regenerated Local Artifacts

These artifacts may be regenerated by the research pipeline but are not all committed:

```text
models/alpha_oos_predictions.csv
models/walkforward_alpha_oos_predictions.csv
models/crypto20_walkforward_alpha_oos_predictions.csv
models/crypto20_walkforward_regime_assignments.csv
models/backtest_curves.csv
models/regime_assignments.csv
models/regime_posteriors.csv
```

They can be useful locally but are too large or too row-level for the public repository.

## Ignored Private or Heavy Artifacts

These must stay out of GitHub:

```text
.env
data/
models/*.pt
models/*.npy
models/*posteriors*.npy
models/*embeddings*.npy
models/*labels*.npy
```

## Versioned Runs

`src/archive_run.py` copies curated artifacts into `runs/<run_id>/` and records SHA-256 hashes in `artifact_manifest.csv`.

Use this for freezing a public research snapshot:

```powershell
python src/archive_run.py --phase phase28_reproduction --source-ref HEAD --notes "Curated reproduction snapshot."
```

## Dashboard Versus Research Artifacts

The Streamlit dashboard uses curated committed artifacts so it can deploy without raw data or model weights. The full research pipeline requires the local research environment and data store.
