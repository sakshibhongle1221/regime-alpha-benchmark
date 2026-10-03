# Crypto-20 Development Data Freeze

## Status

`crypto20-development-v1` is an immutable **development-observed** dataset snapshot. It is not an untouched final test and is not claimed to be a historical top-liquidity universe.

## Frozen Scope

- Assets: 20
- Feature context start: `2024-08-11T03:30:00+05:30`
- Prediction start: `2024-08-13T14:30:00+05:30`
- Prediction end: `2026-06-14T02:30:00+05:30`
- Prediction rows: 321,380
- Walk-forward folds: 16

## Provenance

Existing manually ranked Crypto-20 design universe; frozen without using repaired model outcomes. This is not claimed to be a historical point-in-time top-liquidity universe.

## Integrity

- Database SHA-256: `ef8eb3acdc1eced9cfb72b70f81fb61383cdb2de5167f5a89a2b6c82c7672abe`
- Experiment-data SHA-256: `78b5a0292db5140e418ffb77c05c4e6db943b024a890cd8b36969fa65126bd8e`
- Symbol-manifest SHA-256: `900cea1ce14f332abb7e52b652a3b885335a582f6b4a6026dcf45ca6154d7313`
- Fold-calendar SHA-256: `7e438324746b11037df960c9c0a7fdaa183657154eb41ab899446c6895dde57a`

The database SHA-256 is file-level provenance for the original freeze moment. In later phases the shared local database may contain additional data roles, so verification enforces the frozen experiment-data, symbol, fold, universe, protocol, and configuration hashes rather than unrelated database-file growth. Any change to the frozen asset slice, timestamp range, row count, protocol, or experiment hash creates a different dataset version and must not be silently resumed under this freeze ID.
