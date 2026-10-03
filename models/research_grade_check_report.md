# Research Grade Check Report

- Checks: 387
- Failures: 0
- Warnings: 0

| check | status | detail |
|---|---|---|
| freeze_manifest_exists | PASS | models\crypto20_development_freeze_manifest.json |
| freeze_manifest_invariants | PASS | crypto20-development-v1 invariants hold |
| repaired_classical_experiment_results_exists | PASS | models\crypto20_repaired_classical_experiment_results.csv |
| repaired_classical_experiment_results_readable | PASS | rows=4 columns=24 |
| repaired_classical_coverage_exists | PASS | models\crypto20_repaired_classical_coverage.csv |
| repaired_classical_coverage_readable | PASS | rows=64 columns=4 |
| repaired_classical_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; bad_rows={} |
| repaired_classical_coverage_equal_rows | PASS | rows={'global_lgbm': 230400, 'regime_lgbm_hmm': 230400, 'regime_lgbm_kmeans': 230400, 'regime_lgbm_vol_bucket': 230400} |
| crypto20_repaired_fold_local_experiment_results_exists | PASS | models\crypto20_repaired_fold_local_experiment_results.csv |
| crypto20_repaired_fold_local_experiment_results_readable | PASS | rows=8 columns=24 |
| crypto20_repaired_fold_local_coverage_exists | PASS | models\crypto20_repaired_fold_local_encoder_coverage.csv |
| crypto20_repaired_fold_local_coverage_readable | PASS | rows=128 columns=5 |
| crypto20_repaired_fold_local_fold_metrics_exists | PASS | models\crypto20_repaired_fold_local_fold_metrics.csv |
| crypto20_repaired_fold_local_fold_metrics_readable | PASS | rows=128 columns=25 |
| crypto20_repaired_fold_local_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; expected_rows=230400 |
| crypto20_repaired_fold_local_coverage_equal_rows | PASS | rows={'global_lgbm': 230400, 'regime_lgbm_contrastive': 230400, 'regime_lgbm_contrastive_hmm': 230400, 'regime_lgbm_hmm': 230400, 'regime_lgbm_hmm_guided_gmm': 230400, 'regime_lgbm_hmm_guided_hmm': 230400, 'regime_lgbm_kmeans': 230400, 'regime_lgbm_vol_bucket': 230400} |
| crypto20_repaired_fold_local_fold_metric_coverage | PASS | folds={'global_lgbm': 16, 'regime_lgbm_contrastive': 16, 'regime_lgbm_contrastive_hmm': 16, 'regime_lgbm_hmm': 16, 'regime_lgbm_hmm_guided_gmm': 16, 'regime_lgbm_hmm_guided_hmm': 16, 'regime_lgbm_kmeans': 16, 'regime_lgbm_vol_bucket': 16} |
| phase40_repaired_statistical_method_summary_exists | PASS | models\crypto20_repaired_fold_local_statistical_method_summary.csv |
| phase40_repaired_statistical_method_summary_readable | PASS | rows=8 columns=28 |
| phase40_repaired_statistical_claims_exists | PASS | models\crypto20_repaired_fold_local_statistical_claims.csv |
| phase40_repaired_statistical_claims_readable | PASS | rows=54 columns=15 |
| phase40_repaired_statistical_pairwise_tests_exists | PASS | models\crypto20_repaired_fold_local_statistical_pairwise_tests.csv |
| phase40_repaired_statistical_pairwise_tests_readable | PASS | rows=126 columns=22 |
| phase40_repaired_statistical_fold_metrics_exists | PASS | models\crypto20_repaired_fold_local_statistical_fold_metrics.csv |
| phase40_repaired_statistical_fold_metrics_readable | PASS | rows=128 columns=24 |
| phase40_repaired_statistical_asset_metrics_exists | PASS | models\crypto20_repaired_fold_local_statistical_asset_metrics.csv |
| phase40_repaired_statistical_asset_metrics_readable | PASS | rows=160 columns=24 |
| phase40_repaired_statistical_asset_pairwise_tests_exists | PASS | models\crypto20_repaired_fold_local_statistical_asset_pairwise_tests.csv |
| phase40_repaired_statistical_asset_pairwise_tests_readable | PASS | rows=108 columns=18 |
| phase40_repaired_statistical_test_summary_exists | PASS | models\crypto20_repaired_fold_local_statistical_test_summary.csv |
| phase40_repaired_statistical_test_summary_readable | PASS | rows=54 columns=9 |
| phase40_repaired_statistical_multiple_testing_exists | PASS | models\crypto20_repaired_fold_local_statistical_multiple_testing.csv |
| phase40_repaired_statistical_multiple_testing_readable | PASS | rows=126 columns=34 |
| phase40_repaired_statistical_sharpe_diagnostics_exists | PASS | models\crypto20_repaired_fold_local_statistical_sharpe_diagnostics.csv |
| phase40_repaired_statistical_sharpe_diagnostics_readable | PASS | rows=8 columns=12 |
| phase40_repaired_statistical_ic_confidence_intervals_png | PASS | models\crypto20_repaired_fold_local_statistical_ic_confidence_intervals.png |
| phase40_repaired_statistical_multiple_testing_png | PASS | models\crypto20_repaired_fold_local_statistical_multiple_testing.png |
| phase40_repaired_statistical_sharpe_diagnostics_png | PASS | models\crypto20_repaired_fold_local_statistical_sharpe_diagnostics.png |
| phase40_repaired_statistical_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; bad_folds={}; bad_rows={} |
| phase40_repaired_statistical_no_corrected_alpha_claim | PASS | missing_cols=[]; corrected_ic_sharpe_claims=0 |
| phase40_repaired_statistical_reference_methods | PASS | references=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans'] |
| phase41_config_exists | PASS | configs\phase41_bounded_candidates_v1.json |
| phase41_config_guardrails | PASS | inner-validation-only guardrails present |
| phase41_candidate_registry_exists | PASS | models\phase41_candidate_registry.csv |
| phase41_candidate_registry_readable | PASS | rows=5 columns=8 |
| phase41_selection_rules_exists | PASS | models\phase41_selection_rules.csv |
| phase41_selection_rules_readable | PASS | rows=5 columns=4 |
| phase41_candidate_registry_guardrails | PASS | families=['execution_control', 'probability_calibration', 'soft_regime_gating']; scopes=['inner_validation_only']; threshold_deferred=True |
| phase41_selection_rules_guardrails | PASS | mandatory Phase 41 rules present |
| phase41_report_exists | PASS | reports\phase41_bounded_improvement_protocol.md |
| phase41_report_guardrails | PASS | Phase 41 report guardrails present |
| phase41b_runner_exists | PASS | src\phase41_inner_validation_candidates.py |
| phase41b_runner_tests_exist | PASS | tests\test_phase41_inner_validation_candidates.py |
| phase41b_runner_ps1_exists | PASS | scripts\phases\run_phase41_inner_validation_candidates.ps1 |
| phase41b_runner_sh_exists | PASS | scripts\phases\run_phase41_inner_validation_candidates.sh |
| phase41b_full_experiment_results_exists | PASS | models\phase41_classical_experiment_results.csv |
| phase41b_full_experiment_results_readable | PASS | rows=4 columns=23 |
| phase41b_full_fold_metrics_exists | PASS | models\phase41_classical_fold_metrics.csv |
| phase41b_full_fold_metrics_readable | PASS | rows=64 columns=24 |
| phase41b_full_selected_candidates_exists | PASS | models\phase41_classical_selected_candidates.csv |
| phase41b_full_selected_candidates_readable | PASS | rows=64 columns=9 |
| phase41b_full_inner_candidate_results_exists | PASS | models\phase41_classical_inner_candidate_results.csv |
| phase41b_full_inner_candidate_results_readable | PASS | rows=1024 columns=9 |
| phase41b_full_statistical_claims_exists | PASS | models\phase41_classical_statistical_claims.csv |
| phase41b_full_statistical_claims_readable | PASS | rows=18 columns=15 |
| phase41b_full_statistical_method_summary_exists | PASS | models\phase41_classical_statistical_method_summary.csv |
| phase41b_full_statistical_method_summary_readable | PASS | rows=4 columns=28 |
| phase41b_full_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; bad_rows={} |
| phase41b_full_fold_coverage | PASS | folds={'global_lgbm': 16, 'regime_lgbm_hmm': 16, 'regime_lgbm_kmeans': 16, 'regime_lgbm_vol_bucket': 16} |
| phase41b_no_corrected_alpha_claim | PASS | corrected_ic_sharpe_claims=0 |
| phase41b_candidate_scope_guardrail | PASS | unexpected=[]; deferred_present=False |
| phase41b_statistical_methods | PASS | methods=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket'] |
| phase41b_full_report_exists | PASS | reports\phase41_inner_validation_candidate_run.md |
| phase41b_full_report_guardrails | PASS | Phase 41B full report guardrails present |
| phase42_runner_exists | PASS | src\phase42_interpretation_execution.py |
| phase42_tests_exist | PASS | tests\test_phase42_interpretation_execution.py |
| phase42_runner_ps1_exists | PASS | scripts\phases\run_phase42_interpretation_execution.ps1 |
| phase42_runner_sh_exists | PASS | scripts\phases\run_phase42_interpretation_execution.sh |
| phase42_execution_stress_results_exists | PASS | models\phase42_execution_stress_results.csv |
| phase42_execution_stress_results_readable | PASS | rows=192 columns=13 |
| phase42_execution_stress_summary_exists | PASS | models\phase42_execution_stress_summary.csv |
| phase42_execution_stress_summary_readable | PASS | rows=12 columns=13 |
| phase42_regime_transition_diagnostics_exists | PASS | models\phase42_regime_transition_diagnostics.csv |
| phase42_regime_transition_diagnostics_readable | PASS | rows=7 columns=10 |
| phase42_stable_transition_alpha_exists | PASS | models\phase42_stable_transition_alpha.csv |
| phase42_stable_transition_alpha_readable | PASS | rows=14 columns=8 |
| phase42_cross_asset_alpha_diagnostics_exists | PASS | models\phase42_cross_asset_alpha_diagnostics.csv |
| phase42_cross_asset_alpha_diagnostics_readable | PASS | rows=12 columns=9 |
| phase42_feature_family_diagnostics_exists | PASS | models\phase42_feature_family_diagnostics.csv |
| phase42_feature_family_diagnostics_readable | PASS | rows=6 columns=8 |
| phase42_execution_stress_coverage | PASS | benchmarks=['phase39r_repaired_neural', 'phase41b_classical_candidates']; methods_by_benchmark={'phase39r_repaired_neural': 8, 'phase41b_classical_candidates': 4}; bad_cells={} |
| phase42_execution_summary_guardrail | PASS | rows=12; methods_with_positive_stress_cells=9 |
| phase42_transition_diagnostics_guardrail | PASS | methods=['contrastive', 'contrastive_hmm', 'hmm', 'hmm_guided_gmm', 'hmm_guided_hmm', 'kmeans', 'vol_bucket']; valid_rates=True |
| phase42_stable_transition_alpha_guardrail | PASS | buckets=['stable', 'transition']; rows=14 |
| phase42_cross_asset_alpha_guardrail | PASS | benchmarks=['phase39r_repaired_neural', 'phase41b_classical_candidates']; rows=12 |
| phase42_feature_family_guardrail | PASS | families=['distribution_shape', 'liquidity_volume', 'microstructure', 'momentum', 'technical_state', 'volatility'] |
| phase42_report_exists | PASS | reports\phase42_interpretation_execution_hardening.md |
| phase42_report_guardrails | PASS | Phase 42 report guardrails present |
| phase43a_config_exists | PASS | configs\phase43_locked_holdout_freeze_v1.json |
| phase43a_runner_exists | PASS | src\phase43_locked_holdout_freeze.py |
| phase43a_tests_exist | PASS | tests\test_phase43_locked_holdout_freeze.py |
| phase43a_runner_ps1_exists | PASS | scripts\phases\run_phase43_locked_holdout_freeze.ps1 |
| phase43a_runner_sh_exists | PASS | scripts\phases\run_phase43_locked_holdout_freeze.sh |
| phase43a_locked_candidate_manifest_exists | PASS | models\phase43_locked_candidate_manifest.csv |
| phase43a_locked_candidate_manifest_readable | PASS | rows=20 columns=6 |
| phase43a_locked_claim_rules_exists | PASS | models\phase43_locked_claim_rules.csv |
| phase43a_locked_claim_rules_readable | PASS | rows=9 columns=4 |
| phase43a_locked_holdout_rules_exists | PASS | models\phase43_locked_holdout_rules.csv |
| phase43a_locked_holdout_rules_readable | PASS | rows=17 columns=4 |
| phase43a_config_guardrails | PASS | locked holdout freeze config guardrails present |
| phase43a_manifest_guardrails | PASS | final_rows=1; exclusions=['new_architecture_search', 'new_feature_selection', 'new_label_or_horizon_selection', 'phase41b_probability_calibration', 'phase41b_soft_gating', 'score_threshold_execution_control']; support_hashes=8 |
| phase43a_claim_rules_guardrails | PASS | locked claim rules present |
| phase43a_holdout_rules_guardrails | PASS | rules={'preferred_holdout': 'external_asset_holdout', 'source_universe': 'configs/crypto_universe_candidates.csv', 'selection_rule': 'Select the next pre-ranked quality-eligible USDT spot assets not included in asset_universe_crypto20.csv after ingestion and quality checks, without inspecting model outcomes.', 'minimum_assets': '10', 'fallback_holdout': 'future_temporal_holdout_strictly_after_crypto20_development_endpoint', 'fallback_rule': 'Use only data strictly after 2026-06-15 02:30 Asia/Kolkata, collected and hashed before any model outcome inspection.', 'minimum_hourly_bars': '12000', 'maximum_gap_hours': '6', 'stable_or_synthetic_assets_forbidden': 'True', 'coverage_and_hash_manifest_required': 'True', 'validation': 'same repaired common-calendar purged walk-forward protocol', 'target': 'tb_label_8h', 'horizon_hours': '8', 'transaction_cost_bps': '10', 'candidate_selection_on_holdout': 'forbidden', 'threshold_selection_on_holdout': 'forbidden', 'rerun_after_failure': 'forbidden'} |
| phase43a_report_exists | PASS | reports\phase43_locked_holdout_freeze.md |
| phase43a_report_guardrails | PASS | Phase 43A report guardrails present |
| phase43b_registration_config_exists | PASS | configs\phase43b_locked_holdout_registration_v1.json |
| phase43b_registration_runner_exists | PASS | src\phase43b_locked_holdout_registration.py |
| phase43b_registration_tests_exist | PASS | tests\test_phase43b_locked_holdout_registration.py |
| phase43b_registration_runner_ps1_exists | PASS | scripts\phases\run_phase43b_locked_holdout_registration.ps1 |
| phase43b_registration_runner_sh_exists | PASS | scripts\phases\run_phase43b_locked_holdout_registration.sh |
| phase43b_holdout_candidate_quality_exists | PASS | models\phase43b_holdout_candidate_quality.csv |
| phase43b_holdout_candidate_quality_readable | PASS | rows=75 columns=27 |
| phase43b_registered_holdout_symbols_exists | PASS | models\phase43b_registered_holdout_symbols.csv |
| phase43b_registered_holdout_symbols_readable | PASS | rows=10 columns=14 |
| phase43b_locked_holdout_registration_manifest_exists | PASS | models\phase43b_locked_holdout_registration_manifest.csv |
| phase43b_locked_holdout_registration_manifest_readable | PASS | rows=9 columns=5 |
| phase43b_registration_config_guardrails | PASS | registration-only guardrails present |
| phase43b_registration_quality_guardrails | PASS | missing_cols=[]; development_selected=0 |
| phase43b_registration_manifest_guardrails | PASS | status=registered_ready; selected_count=10; final_candidate=regime_lgbm_hmm_guided_hmm |
| phase43b_registered_symbol_count | PASS | symbols=10; manifest_selected_count=10 |
| phase43b_registration_report_exists | PASS | reports\phase43b_locked_holdout_registration.md |
| phase43b_registration_report_guardrails | PASS | Phase 43B registration report guardrails present |
| phase43b_freeze_runner_exists | PASS | src\phase43b_locked_holdout_freeze.py |
| phase43b_freeze_tests_exist | PASS | tests\test_phase43b_locked_holdout_freeze.py |
| phase43b_adjudication_runner_exists | PASS | src\phase43b_locked_holdout_adjudication.py |
| phase43b_adjudication_tests_exist | PASS | tests\test_phase43b_locked_holdout_adjudication.py |
| phase43b_freeze_report_exists | PASS | reports\phase43b_locked_holdout_data_freeze.md |
| phase43b_eval_report_exists | PASS | reports\phase43b_locked_external_evaluation.md |
| phase43b_adjudication_report_exists | PASS | reports\phase43b_locked_external_adjudication.md |
| phase43b_locked_holdout_symbol_manifest_exists | PASS | models\phase43b_locked_holdout_symbol_manifest.csv |
| phase43b_locked_holdout_symbol_manifest_readable | PASS | rows=10 columns=12 |
| phase43b_locked_holdout_fold_calendar_exists | PASS | models\phase43b_locked_holdout_fold_calendar.csv |
| phase43b_locked_holdout_fold_calendar_readable | PASS | rows=18 columns=11 |
| phase43b_locked_holdout_universe_frozen_exists | PASS | models\phase43b_locked_holdout_universe_frozen.csv |
| phase43b_locked_holdout_universe_frozen_readable | PASS | rows=10 columns=19 |
| phase43b_locked_external_experiment_results_exists | PASS | models\phase43b_locked_external_experiment_results.csv |
| phase43b_locked_external_experiment_results_readable | PASS | rows=8 columns=24 |
| phase43b_locked_external_fold_metrics_exists | PASS | models\phase43b_locked_external_fold_metrics.csv |
| phase43b_locked_external_fold_metrics_readable | PASS | rows=144 columns=25 |
| phase43b_locked_external_encoder_manifest_exists | PASS | models\phase43b_locked_external_encoder_manifest.csv |
| phase43b_locked_external_encoder_manifest_readable | PASS | rows=36 columns=27 |
| phase43b_locked_external_encoder_coverage_exists | PASS | models\phase43b_locked_external_encoder_coverage.csv |
| phase43b_locked_external_encoder_coverage_readable | PASS | rows=144 columns=5 |
| phase43b_locked_external_primary_comparison_exists | PASS | models\phase43b_locked_external_primary_comparison.csv |
| phase43b_locked_external_primary_comparison_readable | PASS | rows=2 columns=17 |
| phase43b_locked_external_claims_exists | PASS | models\phase43b_locked_external_claims.csv |
| phase43b_locked_external_claims_readable | PASS | rows=4 columns=4 |
| phase43b_locked_holdout_freeze_manifest_exists | PASS | models\phase43b_locked_holdout_freeze_manifest.json |
| phase43b_locked_holdout_freeze_guardrails | PASS | locked holdout freeze invariants hold |
| phase43b_symbol_manifest_guardrails | PASS | symbols=10; prediction_rows=[17377] |
| phase43b_fold_calendar_guardrails | PASS | folds=18; min_gap=121.0 |
| phase43b_locked_eval_result_guardrails | PASS | methods=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; row_counts={'global_lgbm': 129600, 'regime_lgbm_contrastive': 129600, 'regime_lgbm_contrastive_hmm': 129600, 'regime_lgbm_hmm': 129600, 'regime_lgbm_hmm_guided_gmm': 129600, 'regime_lgbm_hmm_guided_hmm': 129600, 'regime_lgbm_kmeans': 129600, 'regime_lgbm_vol_bucket': 129600} |
| phase43b_locked_eval_fold_guardrails | PASS | folds={'global_lgbm': 18, 'regime_lgbm_contrastive': 18, 'regime_lgbm_contrastive_hmm': 18, 'regime_lgbm_hmm': 18, 'regime_lgbm_hmm_guided_gmm': 18, 'regime_lgbm_hmm_guided_hmm': 18, 'regime_lgbm_kmeans': 18, 'regime_lgbm_vol_bucket': 18} |
| phase43b_locked_primary_rule_guardrails | PASS | primary frozen relative rule satisfied |
| phase43b_locked_claim_guardrails | PASS | claims={'locked_relative_success_rule': 'satisfied', 'positive_tradable_alpha': 'not_supported', 'candidate_switching_after_holdout': 'forbidden', 'same_holdout_retuning': 'forbidden'} |
| phase43b_locked_adjudication_report_guardrails | PASS | locked adjudication report guardrails present |
| phase44_runner_exists | PASS | src\phase44_paper_readiness_package.py |
| phase44_tests_exist | PASS | tests\test_phase44_paper_readiness_package.py |
| phase44_runner_ps1_exists | PASS | scripts\phases\run_phase44_paper_readiness_package.ps1 |
| phase44_runner_sh_exists | PASS | scripts\phases\run_phase44_paper_readiness_package.sh |
| phase44_report_exists | PASS | reports\phase44_paper_readiness_package.md |
| phase44_reviewer_brief_exists | PASS | reports\phase44_reviewer_brief.md |
| phase44_prepush_hardening_audit_exists | PASS | reports\phase44_prepush_hardening_audit.md |
| phase44_paper_draft_exists | PASS | paper\main.md |
| phase44_paper_protocol_exists | PASS | reports\paper_protocol.md |
| phase44_submission_checklist_exists | PASS | reports\paper_submission_checklist.md |
| phase44_readme_exists | PASS | README.md |
| phase44_paper_evidence_matrix_exists | PASS | models\phase44_paper_evidence_matrix.csv |
| phase44_paper_evidence_matrix_readable | PASS | rows=5 columns=6 |
| phase44_submission_risk_register_exists | PASS | models\phase44_submission_risk_register.csv |
| phase44_submission_risk_register_readable | PASS | rows=6 columns=4 |
| phase44_paper_artifact_map_exists | PASS | reports\paper_artifact_map.csv |
| phase44_paper_artifact_map_readable | PASS | rows=43 columns=4 |
| phase44_evidence_matrix_guardrails | PASS | blocks=['development_statistical_adjudication', 'execution_and_mechanism_diagnostics', 'locked_external_holdout', 'repaired_crypto20_development', 'validation_repair']; data_roles=['development_observed', 'locked_registered_unobserved'] |
| phase44_risk_register_guardrails | PASS | Phase 44 risk register covers critical paper risks |
| phase44_artifact_map_guardrails | PASS | missing_cols=[]; missing_phrases=[] |
| phase44_paper_readiness_package_phase44_guardrails | PASS | Phase 44 claim-control wording present |
| phase44_reviewer_brief_phase44_guardrails | PASS | Phase 44 claim-control wording present |
| phase44_prepush_hardening_audit_phase44_guardrails | PASS | Phase 44 claim-control wording present |
| main_phase44_guardrails | PASS | Phase 44 claim-control wording present |
| paper_protocol_phase44_guardrails | PASS | Phase 44 claim-control wording present |
| paper_submission_checklist_phase44_guardrails | PASS | Phase 44 claim-control wording present |
| README_phase44_guardrails | PASS | Phase 44 claim-control wording present |
| phase45_runner_exists | PASS | src\phase45_venue_manuscript_package.py |
| phase45_tests_exist | PASS | tests\test_phase45_venue_manuscript_package.py |
| phase45_runner_ps1_exists | PASS | scripts\phases\run_phase45_venue_manuscript_package.ps1 |
| phase45_runner_sh_exists | PASS | scripts\phases\run_phase45_venue_manuscript_package.sh |
| phase45_manuscript_exists | PASS | paper\phase45_venue_ready_manuscript.md |
| phase45_package_report_exists | PASS | reports\phase45_venue_manuscript_package.md |
| phase45_reproducibility_appendix_exists | PASS | reports\phase45_reproducibility_appendix.md |
| phase45_submission_checklist_exists | PASS | reports\phase45_submission_checklist.md |
| phase45_external_research_audit_exists | PASS | reports\phase45_external_research_audit.md |
| phase45_table_plan_exists | PASS | models\phase45_table_plan.csv |
| phase45_table_plan_readable | PASS | rows=6 columns=5 |
| phase45_figure_plan_exists | PASS | models\phase45_figure_plan.csv |
| phase45_figure_plan_readable | PASS | rows=4 columns=5 |
| phase45_claim_to_section_map_exists | PASS | models\phase45_claim_to_section_map.csv |
| phase45_claim_to_section_map_readable | PASS | rows=5 columns=5 |
| phase45_venue_requirement_audit_exists | PASS | models\phase45_venue_requirement_audit.csv |
| phase45_venue_requirement_audit_readable | PASS | rows=7 columns=6 |
| phase45_table_plan_guardrails | PASS | Phase 45 table plan covers venue evidence |
| phase45_figure_plan_guardrails | PASS | Phase 45 figure plan covers paper story |
| phase45_claim_map_guardrails | PASS | Phase 45 claim map blocks paper overclaiming |
| phase45_venue_audit_guardrails | PASS | Phase 45 venue audit covers ICAIF/ACM constraints |
| phase45_venue_ready_manuscript_phase45_guardrails | PASS | Phase 45 claim-control wording present |
| phase45_venue_manuscript_package_phase45_guardrails | PASS | Phase 45 claim-control wording present |
| phase45_reproducibility_appendix_phase45_guardrails | PASS | Phase 45 claim-control wording present |
| phase45_submission_checklist_phase45_guardrails | PASS | Phase 45 claim-control wording present |
| phase45_external_research_audit_phase45_guardrails | PASS | Phase 45 claim-control wording present |
| README_phase45_guardrails | PASS | Phase 45 claim-control wording present |
| phase46_runner_exists | PASS | src\phase46_final_research_completion.py |
| phase46_tests_exist | PASS | tests\test_phase46_final_research_completion.py |
| phase46_runner_ps1_exists | PASS | scripts\phases\run_phase46_final_research_completion.ps1 |
| phase46_runner_sh_exists | PASS | scripts\phases\run_phase46_final_research_completion.sh |
| phase46_submission_manuscript_exists | PASS | paper\phase46_submission_manuscript.md |
| phase46_acm_skeleton_exists | PASS | paper\phase46_acm_sigconf_skeleton.tex |
| phase46_final_report_exists | PASS | reports\phase46_final_research_completion.md |
| phase46_submission_audit_exists | PASS | reports\phase46_submission_readiness_audit.md |
| phase46_anonymity_report_exists | PASS | reports\phase46_anonymity_audit.md |
| phase46_reviewer_pack_exists | PASS | reports\phase46_reviewer_response_pack.md |
| phase46_final_todo_exists | PASS | reports\phase46_final_submission_todo.md |
| phase46_section_budget_exists | PASS | models\phase46_section_budget.csv |
| phase46_section_budget_readable | PASS | rows=9 columns=3 |
| phase46_submission_gate_matrix_exists | PASS | models\phase46_submission_gate_matrix.csv |
| phase46_submission_gate_matrix_readable | PASS | rows=10 columns=3 |
| phase46_anonymity_audit_exists | PASS | models\phase46_anonymity_audit.csv |
| phase46_anonymity_audit_readable | PASS | rows=6 columns=4 |
| phase46_final_claim_audit_exists | PASS | models\phase46_final_claim_audit.csv |
| phase46_final_claim_audit_readable | PASS | rows=4 columns=4 |
| phase46_reviewer_objection_matrix_exists | PASS | models\phase46_reviewer_objection_matrix.csv |
| phase46_reviewer_objection_matrix_readable | PASS | rows=5 columns=3 |
| phase46_section_budget_guardrails | PASS | total_budget=8.0; sections=['Abstract', 'Data and Validation', 'Discussion and Limitations', 'Introduction', 'Methods', 'References', 'Related Work', 'Reproducibility', 'Results'] |
| phase46_submission_gate_guardrails | PASS | gates={'venue_scope': 'pass', 'claim_control': 'pass', 'locked_holdout_integrity': 'pass', 'research_gate': 'pass', 'double_blind': 'conditional_pass', 'page_budget': 'conditional_pass', 'acm_template': 'conditional_pass', 'artifact_functionality': 'pass', 'artifact_availability': 'not_claimed', 'final_submission': 'not_yet'} |
| phase46_anonymity_audit_guardrails | PASS | statuses=['pass']; checks=['acknowledgements', 'author_block', 'codex_marker', 'github_link', 'institution_marker', 'personal_name_sakshi'] |
| phase46_claim_audit_guardrails | PASS | Phase 46 final claim audit blocks overclaims |
| phase46_reviewer_objection_guardrails | PASS | Phase 46 reviewer objection pack covers core risks |
| phase46_submission_manuscript_phase46_guardrails | PASS | Phase 46 completion wording present |
| phase46_acm_sigconf_skeleton_phase46_guardrails | PASS | Phase 46 completion wording present |
| phase46_final_research_completion_phase46_guardrails | PASS | Phase 46 completion wording present |
| phase46_submission_readiness_audit_phase46_guardrails | PASS | Phase 46 completion wording present |
| phase46_anonymity_audit_phase46_guardrails | PASS | Phase 46 completion wording present |
| phase46_reviewer_response_pack_phase46_guardrails | PASS | Phase 46 completion wording present |
| phase46_final_submission_todo_phase46_guardrails | PASS | Phase 46 completion wording present |
| README_phase46_guardrails | PASS | Phase 46 completion wording present |
| phase47_runner_exists | PASS | src\phase47_submission_build.py |
| phase47_tests_exist | PASS | tests\test_phase47_submission_build.py |
| phase47_runner_ps1_exists | PASS | scripts\phases\run_phase47_submission_build.ps1 |
| phase47_runner_sh_exists | PASS | scripts\phases\run_phase47_submission_build.sh |
| phase47_submission_draft_exists | PASS | paper\phase47_submission_draft.tex |
| phase47_references_exists | PASS | paper\phase47_references.bib |
| phase47_compiled_pdf_exists | PASS | paper\phase47_submission_draft.pdf |
| phase47_build_report_exists | PASS | reports\phase47_submission_build_report.md |
| phase47_blind_review_report_exists | PASS | reports\phase47_blind_review_hardening.md |
| phase47_gap_list_exists | PASS | reports\phase47_camera_ready_gap_list.md |
| phase47_build_audit_exists | PASS | models\phase47_manuscript_build_audit.csv |
| phase47_build_audit_readable | PASS | rows=10 columns=3 |
| phase47_table_manifest_exists | PASS | models\phase47_table_manifest.csv |
| phase47_table_manifest_readable | PASS | rows=4 columns=6 |
| phase47_figure_manifest_exists | PASS | models\phase47_figure_manifest.csv |
| phase47_figure_manifest_readable | PASS | rows=4 columns=5 |
| phase47_anonymity_audit_exists | PASS | models\phase47_anonymity_source_audit.csv |
| phase47_anonymity_audit_readable | PASS | rows=6 columns=3 |
| phase47_reference_manifest_exists | PASS | models\phase47_reference_manifest.csv |
| phase47_reference_manifest_readable | PASS | rows=8 columns=3 |
| phase47_build_audit_guardrails | PASS | audit={'anonymous_acm_review_mode': 'pass', 'bibliography_present': 'pass', 'tables_present': 'pass', 'no_profitable_strategy_claim': 'pass', 'same_holdout_rescue_blocked': 'pass', 'source_anonymity': 'pass', 'pdf_compilation': 'pass', 'pdf_page_budget': 'pass', 'pdf_warning_review': 'review_required', 'artifact_archive': 'not_claimed'} |
| phase47_table_manifest_guardrails | PASS | tables={'T1': 'included_in_phase47_draft', 'T2': 'included_in_phase47_draft', 'T3': 'included_in_phase47_draft', 'T4': 'appendix_or_reviewer_pack'} |
| phase47_figure_manifest_guardrails | PASS | figures=['F1', 'F2', 'F3', 'F4']; statuses=['needs_final_drawing', 'ready_from_existing_artifacts'] |
| phase47_anonymity_source_guardrails | PASS | anonymity={'author_block': 'pass', 'acknowledgements': 'pass', 'github_link': 'pass', 'personal_name_sakshi': 'pass', 'institution_marker': 'pass', 'tool_marker': 'pass'} |
| phase47_reference_manifest_guardrails | PASS | refs=['acmart', 'chen2020simclr', 'hamilton1989regime', 'lightgbm2017', 'lopezdeprado2018afml', 'oord2018cpc', 'rabiner1989hmm', 'tsfresh2018'] |
| phase47_gitignore_curated_csv_guardrails | PASS | Phase 47 curated CSV artifacts are unignored |
| phase47_submission_draft_phase47_guardrails | PASS | Phase 47 submission-build wording present |
| phase47_submission_build_report_phase47_guardrails | PASS | Phase 47 submission-build wording present |
| phase47_blind_review_hardening_phase47_guardrails | PASS | Phase 47 submission-build wording present |
| phase47_camera_ready_gap_list_phase47_guardrails | PASS | Phase 47 submission-build wording present |
| README_phase47_guardrails | PASS | Phase 47 submission-build wording present |
| phase48_runner_exists | PASS | src\phase48_paper_polish.py |
| phase48_tests_exist | PASS | tests\test_phase48_paper_polish.py |
| phase48_runner_ps1_exists | PASS | scripts\phases\run_phase48_paper_polish.ps1 |
| phase48_runner_sh_exists | PASS | scripts\phases\run_phase48_paper_polish.sh |
| phase48_review_tex_exists | PASS | paper\phase48_review_ready_manuscript.tex |
| phase48_references_exists | PASS | paper\phase48_references.bib |
| phase48_compiled_pdf_exists | PASS | paper\phase48_review_ready_manuscript.pdf |
| phase48_polish_report_exists | PASS | reports\phase48_paper_polish_report.md |
| phase48_reviewer_guide_exists | PASS | reports\phase48_reviewer_reading_guide.md |
| phase48_warning_plan_exists | PASS | reports\phase48_latex_warning_resolution.md |
| phase48_quality_audit_exists | PASS | models\phase48_paper_quality_audit.csv |
| phase48_quality_audit_readable | PASS | rows=6 columns=3 |
| phase48_claim_traceability_exists | PASS | models\phase48_claim_traceability.csv |
| phase48_claim_traceability_readable | PASS | rows=5 columns=5 |
| phase48_warning_audit_exists | PASS | models\phase48_latex_warning_audit.csv |
| phase48_warning_audit_readable | PASS | rows=4 columns=3 |
| phase48_section_budget_exists | PASS | models\phase48_section_budget.csv |
| phase48_section_budget_readable | PASS | rows=9 columns=3 |
| phase48_quality_audit_guardrails | PASS | failures=[]; audit={'anonymous_review_mode': 'pass', 'claim_boundaries': 'pass', 'locked_holdout_boundary': 'pass', 'candidate_switching_boundary': 'pass', 'negative_result_framing': 'pass', 'pdf_warning_state_recorded': 'pass'} |
| phase48_warning_audit_guardrails | PASS | warnings={'pdf_compilation': 'pass', 'build_logs_available': 'pass', 'warning_cleanup': 'review_required', 'page_budget': 'pass'} |
| phase48_claim_trace_guardrails | PASS | claims={'C1': 'allowed', 'C2': 'allowed', 'C3': 'blocked', 'C4': 'blocked', 'C5': 'blocked'} |
| phase48_section_budget_guardrails | PASS | sections=['Abstract', 'Data and Validation Protocol', 'Discussion and Limitations', 'Introduction', 'Methods', 'References', 'Related Work', 'Reproducibility', 'Results'] |
| phase48_gitignore_curated_artifact_guardrails | PASS | Phase 48 curated CSV/PDF artifacts are unignored |
| phase48_review_ready_manuscript_phase48_guardrails | PASS | Phase 48 paper-polish wording present |
| phase48_paper_polish_report_phase48_guardrails | PASS | Phase 48 paper-polish wording present |
| phase48_reviewer_reading_guide_phase48_guardrails | PASS | Phase 48 paper-polish wording present |
| phase48_latex_warning_resolution_phase48_guardrails | PASS | Phase 48 paper-polish wording present |
| README_phase48_guardrails | PASS | Phase 48 paper-polish wording present |
| phase49_runner_exists | PASS | src\phase49_submission_closure.py |
| phase49_tests_exist | PASS | tests\test_phase49_submission_closure.py |
| phase49_runner_ps1_exists | PASS | scripts\phases\run_phase49_submission_closure.ps1 |
| phase49_runner_sh_exists | PASS | scripts\phases\run_phase49_submission_closure.sh |
| phase49_submission_tex_exists | PASS | paper\phase49_submission_ready_manuscript.tex |
| phase49_references_exists | PASS | paper\phase49_references.bib |
| phase49_compiled_pdf_exists | PASS | paper\phase49_submission_ready_manuscript.pdf |
| phase49_closure_report_exists | PASS | reports\phase49_final_submission_closure.md |
| phase49_metadata_report_exists | PASS | reports\phase49_pdf_metadata_audit.md |
| phase49_artifact_guide_exists | PASS | reports\phase49_artifact_release_guide.md |
| phase49_phase49_f1_validation_repair_exists | PASS | paper\figures\phase49_f1_validation_repair.svg |
| phase49_phase49_f1_validation_repair_exists | PASS | paper\figures\phase49_f1_validation_repair.pdf |
| phase49_phase49_f1_validation_repair_exists | PASS | paper\figures\phase49_f1_validation_repair.png |
| phase49_phase49_f2_system_overview_exists | PASS | paper\figures\phase49_f2_system_overview.svg |
| phase49_phase49_f2_system_overview_exists | PASS | paper\figures\phase49_f2_system_overview.pdf |
| phase49_phase49_f2_system_overview_exists | PASS | paper\figures\phase49_f2_system_overview.png |
| phase49_submission_audit_exists | PASS | models\phase49_submission_closure_audit.csv |
| phase49_submission_audit_readable | PASS | rows=6 columns=3 |
| phase49_figure_manifest_exists | PASS | models\phase49_figure_manifest.csv |
| phase49_figure_manifest_readable | PASS | rows=2 columns=6 |
| phase49_citation_audit_exists | PASS | models\phase49_citation_audit.csv |
| phase49_citation_audit_readable | PASS | rows=11 columns=5 |
| phase49_metadata_audit_exists | PASS | models\phase49_pdf_metadata_audit.csv |
| phase49_metadata_audit_readable | PASS | rows=8 columns=3 |
| phase49_artifact_checklist_exists | PASS | models\phase49_artifact_release_checklist.csv |
| phase49_artifact_checklist_readable | PASS | rows=5 columns=3 |
| phase49_latex_warning_audit_exists | PASS | models\phase49_latex_warning_audit.csv |
| phase49_latex_warning_audit_readable | PASS | rows=4 columns=3 |
| phase49_submission_audit_guardrails | PASS | failures=[]; audit={'figures_f1_f2_generated': 'pass', 'citation_expansion': 'pass', 'metadata_anonymity': 'pass', 'artifact_availability_boundary': 'pass', 'latex_warning_state_recorded': 'pass', 'no_model_or_locked_data_change': 'pass'} |
| phase49_figure_manifest_guardrails | PASS | figures=['F1', 'F2']; statuses=['ready_for_visual_review'] |
| phase49_citation_audit_guardrails | PASS | citations=['chen2020simclr', 'cost2022', 'hamilton1989regime', 'lightgbm2017', 'lopezdeprado2018afml', 'oord2018cpc', 'rabiner1989hmm', 'tfc2022', 'tnc2021', 'ts2vec2022', 'tstcc2021']; statuses=['pass'] |
| phase49_metadata_audit_guardrails | PASS | metadata={'pdf_exists': 'pass', 'pdf_size': 'pass', 'metadata_title': 'pass', 'metadata_author': 'pass', 'metadata_subject': 'pass', 'metadata_creator': 'pass', 'metadata_producer': 'pass', 'metadata_keywords': 'pass'} |
| phase49_artifact_boundary_guardrails | PASS | artifact_checklist={'release_tag': 'pending_human_action', 'archive_target': 'pending_human_action', 'doi': 'not_claimed', 'license': 'pending_human_action', 'double_blind': 'review_required'} |
| phase49_latex_warning_guardrails | PASS | warnings={'pdf_compilation': 'pass', 'build_logs_available': 'pass', 'page_budget': 'pass', 'warning_cleanup': 'review_required'} |
| phase49_gitignore_curated_artifact_guardrails | PASS | Phase 49 curated CSV/PDF artifacts are unignored |
| phase49_submission_ready_manuscript_phase49_guardrails | PASS | Phase 49 submission-closure wording present |
| phase49_final_submission_closure_phase49_guardrails | PASS | Phase 49 submission-closure wording present |
| phase49_pdf_metadata_audit_phase49_guardrails | PASS | Phase 49 submission-closure wording present |
| phase49_artifact_release_guide_phase49_guardrails | PASS | Phase 49 submission-closure wording present |
| README_phase49_guardrails | PASS | Phase 49 submission-closure wording present |
| phase39r_neural_full_v1_run_state_exists | PASS | .tmp\phase39_fold_local\phase39r_neural_full_v1\run_state.json |
| phase39r_neural_full_v1_checkpoint_count | PASS | 16/16 |
| phase39r_neural_full_v1_checkpoint_methods | PASS | observed=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket'] |
| claim_registry_exists | PASS | reports\claim_registry.md |
| claim_registry_claim_control | PASS | required claim-control phrases present |
| phase39r_neural_fold_local_results_exists | PASS | reports\phase39r_neural_fold_local_results.md |
| phase39r_neural_fold_local_results_claim_control | PASS | required claim-control phrases present |
| phase40_repaired_statistical_adjudication_exists | PASS | reports\phase40_repaired_statistical_adjudication.md |
| phase40_repaired_statistical_adjudication_claim_control | PASS | required claim-control phrases present |
| phase41_bounded_improvement_protocol_exists | PASS | reports\phase41_bounded_improvement_protocol.md |
| phase41_bounded_improvement_protocol_claim_control | PASS | required claim-control phrases present |
| phase42_interpretation_execution_hardening_exists | PASS | reports\phase42_interpretation_execution_hardening.md |
| phase42_interpretation_execution_hardening_claim_control | PASS | required claim-control phrases present |
| phase43_locked_holdout_freeze_exists | PASS | reports\phase43_locked_holdout_freeze.md |
| phase43_locked_holdout_freeze_claim_control | PASS | required claim-control phrases present |
| phase43b_locked_holdout_registration_exists | PASS | reports\phase43b_locked_holdout_registration.md |
| phase43b_locked_holdout_registration_claim_control | PASS | required claim-control phrases present |
| phase43b_locked_external_adjudication_exists | PASS | reports\phase43b_locked_external_adjudication.md |
| phase43b_locked_external_adjudication_claim_control | PASS | required claim-control phrases present |
| phase44_paper_readiness_package_exists | PASS | reports\phase44_paper_readiness_package.md |
| phase44_paper_readiness_package_claim_control | PASS | required claim-control phrases present |
| phase45_venue_ready_manuscript_exists | PASS | paper\phase45_venue_ready_manuscript.md |
| phase45_venue_ready_manuscript_claim_control | PASS | required claim-control phrases present |
| phase45_venue_manuscript_package_exists | PASS | reports\phase45_venue_manuscript_package.md |
| phase45_venue_manuscript_package_claim_control | PASS | required claim-control phrases present |
| phase46_final_research_completion_exists | PASS | reports\phase46_final_research_completion.md |
| phase46_final_research_completion_claim_control | PASS | required claim-control phrases present |
| phase46_submission_manuscript_exists | PASS | paper\phase46_submission_manuscript.md |
| phase46_submission_manuscript_claim_control | PASS | required claim-control phrases present |
| main_exists | PASS | paper\main.md |
| main_claim_control | PASS | required claim-control phrases present |
| freeze_verify_command | PASS | returncode=0; last_output=OK: crypto20-development-v1 matches its configuration, database, symbol manifest, and fold calendar. |
| unit_tests_command | PASS | returncode=0; last_output=OK |
| calendar_audit_command | PASS | returncode=0; last_output=OK: 20 symbols share one calendar index and all 16 folds have strict global train/test separation under crypto20-development-v1. |
