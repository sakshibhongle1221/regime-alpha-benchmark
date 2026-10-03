# Phase 34 Crypto-20 Guided Encoder Readiness

Phase 34 is a reviewer-facing gate before running the expensive learned-regime expansion.
It checks whether the Phase 33 raw-feature HMM states provide enough weak-supervision signal
for HMM-guided contrastive learning across the full Crypto-20 universe.

## Current Readiness Result

- Eligible HMM-labeled encoder windows: 348,606
- Directed in-trajectory hard-negative pairs within the configured gap: 6,278,476
- Estimated full 30-epoch CPU training time: 16.77 hours
- Gate recommendation: `run_full_crypto20_guided_encoder`

## Why This Matters

The paper should not jump from a two-asset guided encoder to a multi-asset claim without
showing that the weak-supervision signal scales. This phase checks regime coverage,
positive-anchor availability, hard-negative availability near state boundaries, and compute cost.

## Next Phase

If the gate recommends a prototype, run a stratified Crypto-20 guided encoder prototype first.
If that prototype preserves structural alignment, then promote it to a full 30-epoch run and
compare learned regimes against the Phase 33 frozen classical baseline.
