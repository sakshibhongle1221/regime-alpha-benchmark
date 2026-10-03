# Regime Alpha Benchmark Experiment Manifest

## Purpose

This manifest defines the experiments that belong in the paper track after Phase 23. It prevents new experiments from being added without a clear reason.

## Frozen Baseline

| Item | Value |
|---|---|
| Baseline registry | `models/run_registry.csv` |
| Frozen run ID | `20260522_phase14b_baseline` |
| Baseline purpose | Preserve curated Phase 14B artifacts before later statistical/encoder work |
| Main predictive artifact | `models/walkforward_alpha_oos_predictions.csv` |
| Main result artifact | `models/walkforward_experiment_results.csv` |

## Completed Experiment Families

| Family | Phase | Purpose | Main Artifacts | Paper Role |
|---|---|---|---|---|
| Financial labels | 4/12 | Create directional and triple-barrier labels | `target_distribution.csv`, `target_quality.csv` | Methodology |
| Classical regime baselines | 4/13 | HMM, KMeans, volatility bucket comparisons | `regime_benchmark_summary.csv`, `walkforward_regime_summary.csv` | Baseline |
| Alpha benchmark | 4/13/20 | Compare global/regime-conditioned LightGBM models | `experiment_results.csv`, `walkforward_experiment_results.csv`, `guided_alpha_comparison.csv` | Main results |
| Validation audit | 12 | Leakage, fold, coverage, and artifact checks | `validation_audit.csv`, `fold_audit.csv` | Reproducibility |
| Robustness matrix | 14A/21 | Symbol and horizon robustness | `robustness_results.csv`, `robustness_summary.csv` | Robustness |
| Stress robustness | 14B/21 | Cost, threshold, and market-period stress tests | `robustness_stress_results.csv`, `robustness_stress_summary.csv` | Robustness |
| Statistical tests | 15A/15B | Fold-level tests, multiple-testing correction, PSR | `statistical_test_summary.csv`, `statistical_claims.csv` | Evidence control |
| Regime quality | 16 | Structural regime diagnostics | `regime_quality_summary.csv`, `regime_quality_agreement.csv` | Mechanism |
| Compute plan | 17 | Bound experiment cost | `compute_profile.csv`, `ablation_budget.csv` | Scope control |
| Guided encoder | 18/19B | HMM-guided contrastive representation learning | `guided_encoder_summary.csv`, `guided_encoder_comparison.csv` | Proposed method |
| Related work | 19A | Literature positioning | `related_work.md`, `literature_matrix.csv` | Paper framing |
| Time-frequency prototype | 22A | Cheap augmentation check | `time_frequency_encoder_summary.csv`, `time_frequency_encoder_comparison.csv` | Ablation candidate |
| Interpretability | 23 | Fold-local feature attribution | `feature_importance_by_regime.csv`, `feature_family_summary.csv` | Mechanism and discussion |
| Minimal ablation suite | 25 | Objective, assignment-layer, augmentation, and classical-reference ablations | `ablation_results.csv`, `ablation_summary.csv`, `ablation_heatmap.png` | Mechanism test |
| Paper statistical claim tests | 26 | Map ablation mechanisms to fold-level paper evidence | `paper_claim_tests.csv`, `paper_statistical_summary.csv` | Claim discipline |
| Paper draft skeleton | 27 | Convert the evidence stack into a manuscript scaffold | `paper/main.md`, `paper_artifact_map.csv`, `paper_submission_checklist.md` | Writing and submission planning |
| Reproducibility package | 28 | Document public reproduction commands, environment split, and artifact policy | `reproduce.ps1`, `environment.md`, `artifact_manifest.md`, `reproduction_checklist.md` | Public review readiness |
| Paper prose pass | 29 | Turn the generated scaffold into manuscript-style prose while preserving claim discipline | `paper/main.md` | Paper writing |
| Multi-asset universe and data | 31/32 | Pre-specify Crypto-20/Crypto-50 and build the quality-gated data pipeline | `asset_universe_crypto20.csv`, `crypto20_data_quality.csv` | Generalization protocol |
| Crypto-20 classical benchmark | 33 | Establish multi-asset HMM, KMeans, and volatility references | `crypto20_regime_benchmark_summary.csv` | Multi-asset baseline |
| Crypto-20 guided structure | 34/35 | Gate and run the full guided structural experiment | `crypto20_guided_gate.csv`, `crypto20_guided_encoder_summary.csv` | Structural transfer |
| Crypto-20 downstream alpha | 36 | Run the equal-coverage fold-local downstream comparison | `crypto20_walkforward_experiment_results.csv` | Predictive generalization |
| Crypto-20 statistical adjudication | 37 | Test fold uncertainty, calibration, corrections, and asset heterogeneity | `crypto20_statistical_method_summary.csv`, `crypto20_statistical_claims.csv` | Claim adjudication |
| Crypto-20 validity repair and development freeze | 39R-A/B | Repair global calendar folds and evaluation, then freeze the exact development snapshot | `crypto20_development_freeze_manifest.json`, `crypto20_development_fold_calendar.csv`, `evaluation_protocol.md` | Required validity foundation before repaired baselines |
| Repaired Crypto-20 classical baseline | 39R-C | Rerun global LightGBM and raw-regime LightGBM baselines on the frozen calendar-safe development panel | `crypto20_repaired_classical_experiment_results.csv`, `crypto20_repaired_classical_fold_metrics.csv`, `crypto20_repaired_classical_coverage.csv` | Leakage-safe baseline foundation; does not support positive alpha for the classical ladder |
| Repaired Crypto-20 neural/guided baseline | 39R-D | Rerun vanilla contrastive, contrastive-HMM, and HMM-guided fold-local encoders on the same frozen calendar-safe panel | `crypto20_repaired_fold_local_experiment_results.csv`, `crypto20_repaired_fold_local_fold_metrics.csv`, `phase39r_neural_fold_local_results.md` | Leakage-safe neural/guided development benchmark; does not support robust positive alpha |
| Research-grade regression gate | 39R-QA | Add a repeatable artifact/full check loop for future feature changes | `research_grade_check_report.csv`, `research_grade_check_report.md`, `src/research_grade_checks.py` | Reproducibility and regression control |
| Repaired statistical adjudication | 40 | Compare repaired Phase 39R methods with paired fold tests, secondary asset diagnostics, NLL diagnostics, multiple-testing correction, and PSR | `crypto20_repaired_fold_local_statistical_method_summary.csv`, `crypto20_repaired_fold_local_statistical_claims.csv`, `phase40_repaired_statistical_adjudication.md` | No corrected IC/Sharpe superiority or positive-alpha claim is supported; guides bounded development-only next steps |
| Bounded calibration and soft-gating protocol | 41 | Register probability calibration, prior blending, posterior-temperature, shrinkage, and score-threshold candidates with inner-validation-only selection rules | `phase41_candidate_registry.csv`, `phase41_selection_rules.csv`, `phase41_bounded_improvement_protocol.md` | Improvement infrastructure only; score-threshold execution control is registered but deferred; no performance claim and no Phase 40 outer-test tuning |
| Inner-validation candidate runner | 41B scaffold | Implement the global/classical candidate runner with inner-validation selection and outer-fold evaluation | `phase41_inner_validation_candidates.py`, `scripts/phases/run_phase41_inner_validation_candidates.ps1`, `test_phase41_inner_validation_candidates.py` | Code scaffold and smoke check only |
| Full inner-validation global/classical candidate run | 41B full | Run bounded probability-calibration and soft-gating candidates selected only by inner validation across all 16 folds; deferred score-threshold candidates are not included | `phase41_classical_experiment_results.csv`, `phase41_classical_statistical_claims.csv`, `phase41_inner_validation_candidate_run.md` | Controlled negative result; no corrected IC/Sharpe dominance or positive-alpha claim is supported |
| Cross-asset interpretation and execution hardening | 42 | Diagnose why repaired alpha remains weak using execution stress, transition/stable alpha, cross-asset fragility, and feature-family target alignment | `phase42_execution_stress_summary.csv`, `phase42_regime_transition_diagnostics.csv`, `phase42_cross_asset_alpha_diagnostics.csv`, `phase42_interpretation_execution_hardening.md` | Diagnostic explanation only; no tuning, no locked-test claim, and no tradability claim |
| Locked holdout freeze | 43A | Freeze exactly one final guided-HMM mechanism candidate, holdout selection rule, claim rules, and no-retuning policy before any locked outcome is inspected | `phase43_locked_candidate_manifest.csv`, `phase43_locked_claim_rules.csv`, `phase43_locked_holdout_rules.csv`, `phase43_locked_holdout_freeze.md` | Candidate-freeze gate complete; locked evaluation itself is still not run |
| Locked holdout registration | 43B-register | Register locked external holdout membership readiness using only quality/coverage checks before any model outcome is evaluated | `phase43b_holdout_candidate_quality.csv`, `phase43b_registered_holdout_symbols.csv`, `phase43b_locked_holdout_registration_manifest.csv`, `phase43b_locked_holdout_registration.md` | Complete; 10 external assets registered before model evaluation |
| Locked external holdout evaluation | 43B-eval | Evaluate the Phase 43A frozen candidate and references once on the registered external holdout | `phase43b_locked_external_experiment_results.csv`, `phase43b_locked_external_fold_metrics.csv`, `phase43b_locked_external_adjudication.md` | Complete; frozen relative IC/Sharpe rule is satisfied, but profitable/tradable alpha is not supported |
| Paper-readiness evidence package | 44 | Convert the repaired and locked evidence into paper-facing claims, risks, reviewer responses, and manuscript prose without model rescue | `phase44_paper_evidence_matrix.csv`, `phase44_submission_risk_register.csv`, `phase44_paper_readiness_package.md`, `phase44_reviewer_brief.md`, `paper/main.md` | Complete; paper path is limited locked relative support plus no tradable-alpha claim |
| Venue-ready manuscript package | 45 | Convert the evidence package into a venue-facing manuscript source, table/figure plan, venue requirement audit, reproducibility appendix, and submission checklist without model rescue | `phase45_venue_ready_manuscript.md`, `phase45_venue_manuscript_package.md`, `phase45_external_research_audit.md`, `phase45_reproducibility_appendix.md`, `phase45_table_plan.csv`, `phase45_claim_to_section_map.csv`, `phase45_venue_requirement_audit.csv` | Complete; next work is final formatting, citations, figures, anonymity audit, artifact archive, and reviewer polish |
| Final research completion package | 46 | Mark the repository evidence package research-complete and create final paper-formatting, anonymity, reviewer-response, claim-audit, and submission-gate artifacts without model rescue | `phase46_submission_manuscript.md`, `phase46_acm_sigconf_skeleton.tex`, `phase46_final_research_completion.md`, `phase46_submission_gate_matrix.csv`, `phase46_final_claim_audit.csv` | Complete as repository evidence package; external submission still requires compiled PDF, citations, figures, final anonymity audit, and artifact archive/DOI decision |
| Submission manuscript build | 47 | Convert the Phase 46 package into a build-oriented anonymous LaTeX draft, references file, compiled local PDF, table/figure manifests, source anonymity audit, build audit, and submission gap list without model rescue | `phase47_submission_draft.tex`, `phase47_references.bib`, `phase47_submission_draft.pdf`, `phase47_submission_build_report.md`, `phase47_manuscript_build_audit.csv`, `phase47_anonymity_source_audit.csv` | Complete as a compiled draft package; external submission still requires venue-template verification, warning cleanup, final figures/citations, PDF/metadata anonymity audit, artifact archive/DOI decision, and human review |
| Review-ready paper polish | 48 | Polish the compiled manuscript into a review-ready package with claim traceability, PDF warning audit, section budget, and reviewer reading guide without model rescue | `phase48_review_ready_manuscript.tex`, `phase48_references.bib`, `phase48_review_ready_manuscript.pdf`, `phase48_paper_quality_audit.csv`, `phase48_claim_traceability.csv`, `phase48_reviewer_reading_guide.md` | Complete as a review-ready draft package; external submission still requires final venue-template verification, PDF/metadata anonymity audit, artifact archive/DOI decision, advisor/human review, and final submission mechanics |
| Final submission closure | 49 | Add final F1/F2 figures, expanded time-series citations, PDF metadata audit, LaTeX warning-state audit, and artifact-release DOI boundary without model rescue | `phase49_submission_ready_manuscript.tex`, `phase49_references.bib`, `phase49_submission_ready_manuscript.pdf`, `phase49_final_submission_closure.md`, `phase49_submission_closure_audit.csv`, `phase49_pdf_metadata_audit.csv`, `phase49_latex_warning_audit.csv` | Complete as repository-side submission closure; external submission still requires human/advisor visual layout review, active venue-template verification, and artifact DOI decision under double-blind rules |

## Future Experiment Queue

| Priority | Phase | Experiment | Gate | Expected Output |
|---:|---|---|---|---|
| 1 | 38 | Research-control reset | Required before new model development | `phase38_master_protocol.md`, `data_role_registry.csv`, `experiment_ledger.csv`, `publication_acceptance_gates.md` |
| 2 | 44 | Manuscript result package | Phase 43B locked result is complete and adjudicated | Complete; converted into Phase 45 venue package |
| 3 | 45 | ICAIF/ACM-style venue package | Required before external submission | Complete; manuscript source, table/figure plan, reproducibility appendix, and claim map exist |
| 4 | 46 | Final research completion package | Required before external submission | Complete; final LaTeX/PDF, references, figures, anonymity audit, artifact DOI/archive decision, and advisor/reviewer feedback remain human-facing submission tasks |
| 5 | 47 | Submission manuscript build | Required before external submission | Complete as compiled draft package; venue-template verification/warning cleanup/final figures/final citations/PDF anonymity/artifact archive/human review remain |
| 6 | 48 | Review-ready paper polish | Required before external submission | Complete as review-ready package; final venue-template verification, PDF/metadata anonymity audit, artifact archive/DOI decision, and human/advisor review remain |
| 7 | 49 | Final submission closure | Required before external submission | Complete as repository-side closure; remaining work is human/advisor visual review, venue-template verification, and DOI/archive decision |

## Minimal Ablation Definition

Phase 25 includes only the ablations needed to support or falsify the current paper mechanism:

| Ablation | Question |
|---|---|
| Vanilla contrastive vs HMM-guided contrastive | Does weak HMM supervision improve learned structure? |
| GMM vs HMM assignment on embeddings | Does sequential assignment matter after representation learning? |
| Time-only vs time-frequency guided encoder | Does the FFT view improve or dilute the guided representation? |
| Encoder depth if compute allows | Is the result sensitive to a minimal architecture change? |

The completed Phase 25 table supports the assignment-layer mechanism most strongly and does not justify expanding time-frequency downstream compute yet. Phase 26 preserves the paper-safe read: HMM assignment is the strongest supported mechanism, while guided-HMM versus raw-feature HMM remains directional rather than statistically decisive. Hard-negative mining remains optional unless the formal paper draft exposes a mechanism gap.

## Submission Readiness Checklist

The project is submission-ready only when:

1. Phase 38 control files exist and classify all inspected evidence as development-observed.
2. Phase 25 ablation artifacts exist and are discussed.
3. Phase 26 refreshed statistical evidence exists.
4. Phase 27 paper skeleton and artifact map exist.
5. Phase 28 reproduction commands and artifact policy docs exist.
6. Phase 29 prose pass preserves the claims allowed by `reports/claim_registry.md`.
7. The paper draft uses only claims allowed by `reports/claim_registry.md`.
8. The validation audit passes with no critical failures.
9. Reproduction commands are documented.
10. The learned encoder and all downstream decisions are fully fold-local.
11. Missing vanilla learned Crypto-20 baselines are present with equal coverage.
12. One candidate is frozen before locked external evaluation.
13. The final paper follows the current venue page, anonymity, citation, and self-containment rules.
14. The Phase 48 claim trace and PDF warning audit are reviewed before external submission.
15. The Phase 49 figures, citation audit, PDF metadata audit, LaTeX warning-state audit, and artifact DOI boundary are reviewed before external submission.
