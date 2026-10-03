# Phase 43B Locked Holdout Data Freeze

## Status

The Phase 43B external holdout is frozen before model evaluation.

- Freeze ID: `phase43b-locked-external-holdout-v1`
- Registration ID: `phase43b-locked-holdout-registration-v1`
- Data role: `locked_registered_unobserved`
- Symbols: 10
- Prediction rows: 173770
- Fold count: 18
- Experiment-data hash: `3d09fa74881cf04edb472f6706b2b94b23237b547ef1261ac737b7481ed4dc36`

No predictions, alpha metrics, method rankings, or threshold choices are used to create this freeze.

## Frozen Symbols

| Order | Symbol | Prediction rows | Start | End |
|---:|---|---:|---|---|
| 1 | `FILUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 2 | `ARBUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 3 | `OPUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 4 | `INJUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 5 | `ATOMUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 6 | `XLMUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 7 | `ALGOUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 8 | `AAVEUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 9 | `RUNEUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |
| 10 | `FETUSDT` | 17377 | 2024-07-05T11:30:00+05:30 | 2026-06-29T11:30:00+05:30 |

## Fold Calendar

- Earliest test: `2025-01-04T00:30:00+05:30`
- Latest test: `2026-06-27T23:30:00+05:30`
- Minimum calendar gap hours: 121.0

## Interpretation

This artifact only freezes the locked-holdout data. The next step is the one-shot frozen evaluation. If that evaluation fails, the result must be reported without same-holdout tuning.
