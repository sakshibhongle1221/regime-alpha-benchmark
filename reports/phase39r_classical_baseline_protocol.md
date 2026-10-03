# Phase 39R-C Repaired Classical Baseline

## Purpose

This gate evaluates the repaired pipeline before any neural encoder is retrained. It is bound to `crypto20-development-v1` and compares only:

1. global LightGBM,
2. raw-feature HMM regimes,
3. raw-feature KMeans regimes,
4. volatility-bucket regimes.

## Required Controls

- one shared Crypto-20 timestamp panel,
- strict pooled train/test calendar separation,
- frozen asset order, rows, database hash, experiment-data hash, and fold calendar,
- fold-local regime fitting,
- equal prediction coverage,
- non-overlapping eight-hour portfolio returns,
- mean per-asset IC as primary IC,
- resume-safe per-fold checkpoints.

## Smoke Status

The isolated one-fold smoke passed with 85,220 training rows and 14,400 test rows per method. The freeze ID and calendar timestamps were recorded, all four methods had equal coverage, repaired metrics were generated, and checkpoint resume reproduced the same compact outputs. One-fold metrics are code-path diagnostics only and are not scientific performance evidence.

## Full-Run Gate

The 16-fold classical development run is complete on `crypto20-development-v1`.

Artifacts:

- `models/crypto20_repaired_classical_experiment_results.csv`
- `models/crypto20_repaired_classical_fold_metrics.csv`
- `models/crypto20_repaired_classical_coverage.csv`
- `models/crypto20_repaired_classical_manifest.csv`
- `models/crypto20_repaired_classical_implementations.csv`

Run metadata:

- completed folds: 16
- methods: global LightGBM, raw-feature HMM regimes, raw-feature KMeans regimes, volatility-bucket regimes
- test rows per method: 230,400
- non-overlapping return observations: 1,440
- freeze ID: `crypto20-development-v1`
- checkpoint run finalized from saved checkpoints after a timezone-only aggregation repair

## Full-Run Result

| Method | Mean asset IC | Cross-sectional IC | Sharpe | Total return | Interpretation |
|---|---:|---:|---:|---:|---|
| global LightGBM | -0.0113 | -0.0048 | -0.088 | -0.0089 | Weak/negative repaired baseline |
| regime LightGBM + HMM | -0.0093 | 0.0016 | -0.260 | -0.0258 | Slightly better IC, worse portfolio metric |
| regime LightGBM + KMeans | -0.0139 | -0.0026 | -0.209 | -0.0188 | Weak/negative |
| regime LightGBM + volatility buckets | -0.0153 | 0.0001 | -0.295 | -0.0267 | Weak/negative |

## Safe Interpretation

This repaired classical gate does not support a positive alpha claim. It is useful because it establishes a leakage-safe, frozen, equal-coverage benchmark before any repaired neural/guided model is rerun.

Allowed claim:

> On the repaired Crypto-20 development snapshot, the classical global and regime-conditioned LightGBM baselines do not show convincing positive predictive alpha under mean per-asset IC and non-overlapping cost-adjusted portfolio diagnostics.

Forbidden claim:

> The guided neural method is better than these baselines.

That claim remains open until the repaired fold-local neural/guided experiment is rerun under the same freeze, calendar, checkpoint, and evaluation protocol.
