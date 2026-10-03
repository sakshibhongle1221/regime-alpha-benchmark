# Regime Alpha Benchmark Compute Budget

## Purpose

This document keeps the research plan compute-aware. The next phases add statistical testing, regime-quality metrics, encoder upgrades, and ablations. Without explicit compute gates, the project can grow into more experiments than a local machine can reasonably run.

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
| Validation audit | about 10 seconds |
| Data health check | a few seconds |

The encoder retraining runtime should be measured before large ablations. Future Phase 17 should add measured timings for one encoder training run on the current machine.

## Mandatory Next Experiments

The next phases should stay small until the baseline is statistically understood.

| Phase | Required Runs | Notes |
|---|---:|---|
| Phase 15B statistical tests | 0 retraining runs | Uses frozen predictions/artifacts |
| Phase 16 regime quality metrics | 0 retraining runs | Uses existing regime assignments |
| Phase 17 compute plan | 1 timing run if needed | Measure encoder retrain cost |
| Phase 18 HMM-guided encoder | 1-2 encoder runs | Minimal proof before ablations |
| Phase 19 time-frequency augmentation | 2-3 encoder runs | Time-only, frequency-only, both |
| Phase 20 hard negatives | 2-3 encoder runs | Random vs in-trajectory vs boundary negatives |
| Phase 21 ablations | capped matrix | Expand only if early results justify it |

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

## Multi-Asset Gate

Phase 23 multi-asset generalization is conditional.

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
Do not expand scope.
Write the paper as a crypto benchmark.
List multi-asset validation as future work.
```

This gate prevents scope creep and avoids spending compute on a generalization claim before the learned encoder has earned it.

## Paper-Safety Rules

- Every major run must be archived with `src/archive_run.py`.
- Every statistical table must reference a `run_id`.
- Every encoder result must record architecture, seed, loss, augmentation, assignment method, and source artifact paths.
- HMM states must be described as proxy/reference regimes, not ground truth.
- Backtest metrics must be described as diagnostics, not deployable trading claims.
