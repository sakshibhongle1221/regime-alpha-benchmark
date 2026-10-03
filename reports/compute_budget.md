# Regime Alpha Benchmark Compute Budget

## Purpose

This document keeps the research plan compute-aware. The next phases add encoder upgrades and ablations on top of the frozen baseline, statistical tests, and regime-quality diagnostics. Without explicit compute gates, the project can grow into more experiments than a local machine can reasonably run.

## Current Environment

| Field | Value |
|---|---|
| Local Python | 3.11.3 |
| OS | Windows |
| Main research dependencies | PyTorch, LightGBM, hmmlearn, scikit-learn, DuckDB |
| Current universe | BTCUSDT + ETHUSDT |
| Current interval | 1h |
| Current feature rows | 34,920 total before target join |
| Current primary OOS rows | 25,920 per method |

The current run registry snapshot is:

```text
runs/20260522_phase14b_baseline/
```

## Observed Runtime Notes

These are practical local observations, not formal benchmarks:

| Task | Observed Runtime |
|---|---:|
| Phase 14A full symbol/horizon robustness grid | about 30 minutes |
| Phase 14B cost/threshold/period stress grid | under 1 minute |
| Phase 15A/15B statistical tests | under 1 minute |
| Phase 16 regime-quality diagnostics | under 1 minute |
| Phase 17 compute planner | about 20 seconds with 5 profiled CPU steps |
| Phase 18 one-epoch guided encoder smoke run | about 4 minutes |
| Phase 19B 30-epoch guided encoder full run | about 61 minutes |
| Phase 20 guided downstream alpha retest | about 12 minutes |
| Phase 21 guided robustness refresh | about 78 minutes total across chunked robustness runs plus stress refresh |
| Phase 22A 3-epoch time-frequency guided encoder prototype | about 11 minutes |
| Phase 23 fold-local interpretability | about 6 minutes with capped SHAP sampling |
| Phase 24 paper protocol freeze | no model compute |
| Phase 25 minimal ablation suite | under 1 minute; no model retraining |
| Phase 26 paper statistical claim tests | under 1 minute; no model retraining |
| Phase 27 paper skeleton generation | under 1 minute; no model retraining |
| Phase 28 reproducibility smoke check | about 30 seconds; no model retraining |
| Validation audit | about 10 seconds |
| Data health check | a few seconds |

Phase 17 measured the current encoder training cost with a synthetic CPU forward/backward timing profile. This is not a formal hardware benchmark, but it is enough to keep the next experiment queue realistic.

| Metric | Value |
|---|---:|
| Device | CPU |
| Encoder parameters | 139,408 |
| Training windows | 34,798 |
| Batches per epoch | 271 |
| Synthetic step time | 0.739 seconds |
| Estimated epoch time | 3.34 minutes |
| Estimated 30-epoch encoder run | 100.10 minutes |
| Estimated 12-run ablation grid | 21.62 hours |
| Local budget | 24 hours |
| Budget status | green |

## Mandatory Next Experiments

The next phases should stay small until the baseline is statistically understood.

| Phase | Required Runs | Notes |
|---|---:|---|
| Phase 15A statistical tests | 0 retraining runs | Complete; uses frozen predictions/artifacts |
| Phase 15B multiple-testing corrections | 0 retraining runs | Complete; extends Phase 15A tables |
| Phase 16 regime quality metrics | 0 retraining runs | Complete; uses existing regime assignments |
| Phase 17 compute plan | 0 retraining runs | Complete; measured local encoder cost using synthetic timing |
| Phase 18 HMM-guided encoder | 1 smoke encoder run | Complete; validated artifact path |
| Phase 19B full HMM-guided encoder | 1 full encoder run | Complete; 30 epochs, time-only, HMM/GMM assignments |
| Phase 20 guided downstream alpha re-test | 0 encoder runs | Complete; uses Phase 19B guided embeddings in fold-local alpha/statistical benchmark |
| Phase 21 guided robustness refresh | 0 encoder runs | Complete; refreshes symbol/horizon and stress robustness with guided methods |
| Phase 22 time-frequency augmentation | 2-3 encoder runs | Prototype complete; full 30-epoch run remains conditional |
| Phase 23 interpretability | 0 encoder runs | Complete; fold-local LightGBM feature-importance and capped SHAP summaries |
| Phase 24 paper protocol freeze | 0 encoder runs | Complete; freezes hypotheses, claims, experiment queue, and submission gate |
| Phase 25 minimal ablations | 0 encoder runs | Complete; aggregates completed structural and downstream artifacts into a mechanism table |
| Phase 26 statistical evidence refresh | 0 encoder runs | Complete; maps the ablation table to paper-facing claim tests |
| Phase 27 paper draft skeleton | 0 encoder runs | Complete; generates manuscript scaffold, artifact map, and submission checklist |
| Phase 28 reproducibility package | 0 encoder runs | Complete; adds smoke/full/dashboard reproduction modes and reviewer-facing artifact policy docs |
| Phase 29 paper prose pass | 0 encoder runs | Complete; turns the scaffold into polished submission-style prose |
| Phase 30 venue formatting and citation pass | 0 encoder runs | Next; convert the Markdown draft into the target venue format |

## Initial Ablation Cap

Do not start with a full combinatorial grid. The initial cap should be:

```text
3 losses x 2 augmentations x 2 assignment methods = 12 runs maximum
```

Where:

```text
losses = NT-Xent, InfoNCE-style, HMM-guided
augmentations = time-only, time+frequency
assignment methods = GMM, HMM
```

Encoder depth and hidden-dimension ablations should only happen after the loss/augmentation/assignment choices show promise.

The first three queued runs are:

| Priority | Loss | Augmentation | Assignment | Decision |
|---:|---|---|---|---|
| 1 | `hmm_guided` | `time_only` | `hmm` | complete; best point-estimate downstream method |
| 2 | `hmm_guided` | `time_only` | `gmm` | complete; weak downstream method |
| 3 | `hmm_guided` | `time_frequency` | `hmm` | 3-epoch prototype complete; full run conditional |

The rest of the 12-run grid should stay on hold until one of these runs improves the learned-regime path.

## Phase 18 Smoke Result

The first Phase 18 implementation run used `guided_encoder.py --epochs 1` to validate the full artifact path before launching the longer run. It produced both GMM and HMM assignments from the HMM-guided embedding space.

| Method | Epochs | Silhouette | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|
| `hmm_guided_gmm` | 1 | 0.341 | 0.387 | 0.652 |
| `hmm_guided_hmm` | 1 | 0.353 | 0.389 | 0.620 |

This is a smoke-test result, not a final performance claim. The full Phase 18 run should use 30 epochs before downstream alpha/statistical comparisons.

## Phase 19B Full Guided Run

The full HMM-guided time-only encoder run completed with the standard 30-epoch setting.

| Method | Epochs | Silhouette | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|
| `hmm_guided_gmm` | 30 | 0.384 | 0.609 | 0.759 |
| `hmm_guided_hmm` | 30 | 0.629 | 0.869 | 0.957 |

Observed runtime was about 61 minutes on the current CPU environment, better than the Phase 17 synthetic estimate of about 99 minutes. This justified the Phase 20 downstream test before launching broader time-frequency or architecture ablations.

## Phase 20 Downstream Result

Phase 20 reused the Phase 19B guided embeddings and added no encoder retraining cost. The fold-local alpha retest completed in about 12 minutes.

| Method | IC | Sharpe | Drawdown | Total Return | Statistical Read |
|---|---:|---:|---:|---:|---|
| `regime_lgbm_hmm` | 0.0051 | -0.340 | -0.710 | -0.536 | raw-feature sequential reference |
| `regime_lgbm_hmm_guided_gmm` | -0.0092 | -0.976 | -0.900 | -0.854 | guided representation with weak assignment layer |
| `regime_lgbm_hmm_guided_hmm` | 0.0094 | 0.099 | -0.614 | 0.031 | best point estimate, not significant versus raw HMM on fold-level IC |

The result is strong enough to justify one focused next encoder experiment: add the time-frequency view to the guided-HMM path. It is not strong enough to justify multi-asset expansion yet.

## Phase 22A Time-Frequency Prototype

Phase 22A completed a capped 3-epoch time-frequency guided encoder run. The run appends six low-frequency FFT magnitude bands per feature to each 60-bar input window, increasing the encoder input from 22 to 154 features.

Observed runtime was about 11 minutes on the current CPU environment.

| Method | Epochs | Input Features | Silhouette | HMM NMI | HMM Purity | Read |
|---|---:|---:|---:|---:|---:|---|
| `tf_hmm_guided_gmm` | 3 | 154 | 0.326 | 0.504 | 0.682 | stronger than vanilla contrastive, weaker than full guided baseline |
| `tf_hmm_guided_hmm` | 3 | 154 | 0.338 | 0.528 | 0.704 | best Phase 22A structural result |

This prototype does not yet justify a downstream alpha retest. The full 30-epoch time-only guided-HMM run remains the active baseline with `HMM NMI = 0.869` and `HMM purity = 0.957`. A full time-frequency run should be launched only if the project has spare compute after interpretability, or if the paper needs an augmentation ablation.

## Multi-Asset Gate

Multi-asset generalization is conditional and should not replace the Phase 28 reproducibility package.

Proceed to multi-asset only if:

```text
best learned encoder beats raw-feature HMM on the primary predictive metric
and the paired/DM-style test reports p < 0.05.
```

Primary metric:

```text
signal IC or a loss-based predictive score from out-of-sample predictions
```

Secondary metrics:

```text
Sharpe, drawdown, total return, turnover
```

If the gate fails:

```text
Do not expand downstream alpha scope.
Write the paper as a crypto benchmark.
List multi-asset predictive validation as future work.
```

This gate prevents scope creep and avoids spending compute on a generalization claim before the learned encoder has earned it.

This gate applies to downstream alpha claims, not to structural diagnostics. Structural generalization tests may proceed independently if they are clearly labeled as regime-quality or representation-quality experiments and do not authorize any claim that the learned method improves out-of-sample alpha on the expanded universe. In practice, this means a Crypto-20 guided encoder run can be used to test whether the HMM-guided objective transfers structurally, while a separate fold-local Crypto-20 alpha benchmark is still required before making any predictive generalization claim.

Current gate status after Phase 20:

```text
point-estimate improvement: yes
fold-level IC significance versus raw HMM: no, p = 0.801
decision: do not expand downstream alpha claims to multi-asset yet
structural generalization testing: permitted with explicit no-alpha-claim language
```

## Paper-Safety Rules

- Every major run must be archived with `src/archive_run.py`.
- Every statistical table must reference a `run_id`.
- Every encoder result must record architecture, seed, loss, augmentation, assignment method, and source artifact paths.
- HMM states must be described as proxy/reference regimes, not ground truth.
- Backtest metrics must be described as diagnostics, not deployable trading claims.
