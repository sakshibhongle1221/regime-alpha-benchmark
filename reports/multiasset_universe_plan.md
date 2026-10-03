# Multi-Asset Universe Plan

## Purpose

Phase 31 upgrades Regime Alpha Benchmark from a BTC/ETH pilot into a reproducible multi-asset research program. The goal is not to add symbols for appearance. The goal is to define a pre-registered crypto universe protocol before running broad experiments, so the final paper can make generalization claims without looking cherry-picked.

## Research Role

The BTC/ETH benchmark remains the controlled pilot. It isolates the core mechanism: sequential assignment makes learned regimes more useful, and HMM-guided weak supervision improves structural regime alignment.

The multi-asset stage has two separate roles. Structural generalization asks whether the regime-learning objective, assignment layer, and diagnostic framework still behave coherently outside the pilot setting. Predictive generalization asks whether those regimes improve fold-local downstream alpha metrics on the wider universe. The Phase 20 `p=0.801` gate blocks predictive multi-asset claims, but it does not block structural generalization diagnostics when they are labeled honestly.

The final paper should treat BTC/ETH as the controlled pilot, Crypto-20 as structural generalization evidence until its fold-local alpha retest is complete, and Crypto-50 as a future publication-level extension only after compute and evidence gates pass.

## Universe Construction

The candidate list starts from liquid Binance USDT spot pairs and excludes stablecoin pairs, leveraged tokens, and synthetic instruments. Final eligibility is determined only after ingestion and quality checks.

Selection rules:

- Use hourly Binance spot OHLCV data.
- Require continuous history over the study window whenever possible.
- Prefer symbols with at least the configured minimum bar count.
- Reject symbols with excessive missing-bar gaps.
- Rank eligible assets by median quote volume.
- Preserve a full exclusion log so failed assets are visible.

## Two-Stage Expansion

1. Crypto-20 pilot: first scalability checkpoint for data, features, targets, regimes, alpha models, and dashboard artifacts.
2. Crypto-50 final: publication-level generalization experiment after the Crypto-20 pipeline is stable.

## Reviewer-Safe Claim

The paper should not claim universal market generalization after Crypto-50. It can claim broad crypto-universe evidence under a controlled exchange, interval, and feature protocol.

Preferred wording:

> We first establish the mechanism in a controlled BTC/ETH pilot, then test whether the same regime-learning conclusions survive across a pre-specified liquid Binance USDT crypto universe.

## Required Artifacts

- `configs/crypto_universe_candidates.csv`
- `models/asset_universe_crypto20.csv`
- `models/asset_universe_crypto50.csv`
- `models/asset_universe_exclusions.csv`
- `models/asset_universe_summary.csv`

## Decision Gate

The downstream alpha gate and the structural diagnostics gate are intentionally separate.

Structural Crypto-20 experiments may proceed if:

- the universe is pre-specified before results are inspected,
- ingestion and quality gates pass,
- the experiment is framed as regime-quality or representation-quality evidence,
- the paper does not claim multi-asset alpha improvement from structural artifacts alone.

Proceed from Crypto-20 to Crypto-50 only if:

- ingestion and feature generation complete without major gaps,
- every selected symbol has enough target rows for walk-forward folds,
- validation audit passes,
- runtime and memory remain manageable,
- no single symbol family dominates the selected universe.
- either the Crypto-20 downstream alpha retest is statistically/economically strong enough to justify a predictive expansion, or the paper explicitly frames Crypto-50 as a structural-only appendix.

If Crypto-20 fails structurally, the paper should remain a BTC/ETH controlled benchmark and describe multi-asset generalization as future work. If Crypto-20 succeeds structurally but not predictively, the paper may discuss structural transfer while keeping alpha generalization as future work.
