# Phase 33 Crypto-20 Regime Benchmark

## Purpose

Phase 33 tests whether the regime-detection story survives the first multi-asset expansion. It uses the Phase 32 Crypto-20 data gate as input and benchmarks three classical regime methods across all selected assets:

- raw-feature Gaussian HMM,
- raw-feature KMeans,
- volatility quantile buckets.

The learned contrastive/guided encoders are intentionally not retrained in this phase. That comes after the classical multi-asset baseline is frozen.

## Why This Phase Matters

The BTC/ETH pilot showed that sequential consistency matters. Crypto-20 asks whether that mechanism remains visible across a broader liquid crypto universe. A reviewer can now see that we did not jump directly from a two-asset pilot into a guided encoder claim; we first established the expanded data and classical regime baseline.

## Current Result

The current Crypto-20 regime benchmark summary covers 349,786 joined feature/target rows across 20 assets. HMM, KMeans, and volatility buckets all cover the same universe.

Key diagnostic pattern:

- KMeans has the highest silhouette but highly imbalanced regimes.
- HMM has lower silhouette but stronger sequential persistence.
- Volatility buckets are balanced by construction and useful as a simple control.

This supports the paper's mechanism-first framing: cluster separability alone is not enough; temporal structure remains central.

## Artifacts

- `models/crypto20_regime_benchmark_summary.csv`
- `models/crypto20_per_regime_stats.csv`
- `models/crypto20_regime_symbol_summary.csv`
- `models/crypto20_transition_matrix_hmm.png`
- `models/crypto20_transition_matrix_kmeans.png`
- `models/crypto20_transition_matrix_vol_bucket.png`

The row-level `models/crypto20_regime_assignments.csv` is generated locally for analysis but intentionally ignored by Git.

## Gate To Next Phase

Proceed to Crypto-20 learned-regime experiments only if:

- all methods cover the same row universe,
- all methods cover all 20 selected assets,
- HMM uses `hmmlearn_gaussian_hmm` rather than fallback,
- no method collapses to fewer than four regimes,
- compact artifacts are regenerated and pass validation audit.
