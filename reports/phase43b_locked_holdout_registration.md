# Phase 43B Locked Holdout Registration

## Status

Phase 43B registration checks whether the external locked holdout is ready before any frozen-model outcome is evaluated.

- Registration ID: `phase43b-locked-holdout-registration-v1`
- Parent freeze: `phase43-locked-holdout-freeze-v1`
- Data role: `locked_unobserved_registration_only`
- Registration status: `registered_ready`
- Selected locked assets: 10
- Minimum required assets: 10
- Missing assets before evaluation: 0

No model predictions, alpha metrics, method rankings, threshold search, or locked-holdout performance outcomes are read in this phase.

## Registered Symbols

| Design rank | Symbol | OHLCV rows | Feature rows | Target rows |
|---:|---|---:|---:|---:|
| 21 | `FILUSDT` | 17520 | 17460 | 17436 |
| 22 | `ARBUSDT` | 17520 | 17460 | 17436 |
| 23 | `OPUSDT` | 17520 | 17460 | 17436 |
| 24 | `INJUSDT` | 17520 | 17460 | 17436 |
| 25 | `ATOMUSDT` | 17520 | 17460 | 17436 |
| 26 | `XLMUSDT` | 17520 | 17460 | 17436 |
| 27 | `ALGOUSDT` | 17520 | 17460 | 17436 |
| 28 | `AAVEUSDT` | 17520 | 17460 | 17436 |
| 30 | `RUNEUSDT` | 17520 | 17460 | 17436 |
| 31 | `FETUSDT` | 17520 | 17460 | 17436 |

## First External Candidates Still Not Registered

These rows explain why the local machine is or is not ready for locked evaluation. This is a data-readiness table, not a model-performance table.

| Design rank | Symbol | OHLCV rows | Feature rows | Target rows | Reason |
|---:|---|---:|---:|---:|---|
| 29 | `MKRUSDT` | 10604 | 10544 | 10520 | insufficient_ohlcv |
| 41 | `SUIUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 42 | `PEPEUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 43 | `FLOKIUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 44 | `RENDERUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 45 | `TAOUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 46 | `ENAUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 47 | `ORDIUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 48 | `ARUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 49 | `SANDUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 50 | `MANAUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 51 | `AXSUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 52 | `APEUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 53 | `CHZUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |
| 54 | `ENSUSDT` | 0 | 0 | 0 | insufficient_ohlcv;missing_features;missing_targets |

## Interpretation

If status is `registered_ready`, the next valid action is one locked evaluation of the Phase 43A frozen candidate.

If status is `blocked_not_ready`, the next valid action is to ingest/build feature and target data for the next pre-ranked external candidates, rerun this registration, and only then evaluate.

Forbidden wording:

```text
Phase 43B registration proves generalization.
Phase 43B registration selected assets using model performance.
Phase 43B registration allows retrying the locked evaluation after failure.
```
