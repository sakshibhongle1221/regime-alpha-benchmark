# Phase 32 Crypto-20 Data Pipeline And Quality Gate

## Purpose

Phase 32 prepares and executes the data layer for the first multi-asset generalization run. It does not claim a Crypto-20 modeling result yet. It makes the ingestion, feature, target, and data-check scripts resolve symbols from the Phase 31 universe artifacts, then records whether the selected assets pass the data quality gate.

## Pipeline Commands

```powershell
python src/multiasset_universe.py
python src/ingestion.py --universe crypto20
python src/features.py --universe crypto20
python src/targets.py --universe crypto20 --artifact-prefix crypto20_
python src/check.py --universe crypto20
python src/crypto20_quality_gate.py --universe crypto20
```

## Current Expected State

Before Crypto-20 ingestion, only BTCUSDT and ETHUSDT should be quality-eligible because they are the only locally ingested pilot assets. The other Crypto-20 symbols should appear as missing or pending. That is not a failure; it is the correct pre-ingestion checkpoint.

After Crypto-20 ingestion and feature generation, the expected state changes:

- all selected Crypto-20 symbols should have OHLCV rows,
- all selected Crypto-20 symbols should have feature rows,
- target rows should exist for every selected symbol,
- `python src/crypto20_quality_gate.py --universe crypto20` should record which selected symbols pass local coverage, feature, target, and gap checks.

## Completion Gate

Consider combined Phase 32 complete only if:

- at least 18 of 20 Crypto-20 assets pass quality gates,
- no selected symbol has large unexplained hourly gaps,
- feature and target rows are available for every eligible symbol,
- the validation audit still passes for BTC/ETH pilot artifacts,
- the Crypto-20 data-quality artifact clearly records any failed symbols.

If fewer than 18 assets pass, the correct response is not to hide the failure. Re-rank the universe using the same pre-specified candidate list and record exclusions transparently.
