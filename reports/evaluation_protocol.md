# Predictive And Portfolio Evaluation Protocol

## Status

This protocol replaces the earlier pooled-IC and overlapping-return evaluation. Historical metrics generated under the old implementation remain development history and must not be compared directly with repaired-run metrics.

## Information Coefficient

The primary multi-asset IC is the mean of per-asset time-series correlations between score and forward return. The artifact also reports:

- median per-asset time-series IC,
- mean and median per-timestamp cross-sectional IC when at least three assets are present,
- pooled asset-time correlation as a secondary diagnostic only,
- the number of valid asset and timestamp correlations.

The generic `IC` column means `mean_asset_IC` in all newly generated artifacts.

## Portfolio Returns

The primary portfolio diagnostic uses one pre-specified non-overlapping UTC rebalance grid:

- prediction horizon: eight hours for the primary target,
- rebalance times: UTC epoch-hour modulo eight equals zero,
- signal execution: one-hour lag after feature observation,
- positions reset at every outer walk-forward fold,
- transaction costs apply to changes in executed position,
- assets are equally weighted at each rebalance timestamp,
- Sharpe annualization uses `sqrt(8760 / horizon_hours)` on non-overlapping horizon returns.

The seven alternative eight-hour offsets are sensitivity checks only. They may not be searched to select the best reported result.

## Interpretation

IC, calibration, and portfolio returns answer different questions. No method is considered superior unless the pre-specified primary comparison is supported at the fold level after correction. Sharpe and total return remain research diagnostics, not live-trading or profitability claims.
