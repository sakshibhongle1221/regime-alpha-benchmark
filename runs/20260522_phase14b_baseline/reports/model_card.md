# Regime Alpha Benchmark Model Card

## Frozen Baseline

| Field | Value |
|---|---|
| Frozen run id | `20260522_phase14b_baseline` |
| Source ref | `v1.3-phase14b` |
| Source commit | `f0902b21057d3dc464ce9e31d6be718a70531c63` |
| Archive manifest | `runs/20260522_phase14b_baseline/manifest.json` |
| Artifact count | 39 curated artifacts |
| Missing artifacts | 0 |

This model card documents the current Phase 14B baseline before statistical testing and encoder upgrades. The frozen run is the reference point for later significance tests, regime-quality metrics, encoder changes, and ablations.

## Data

| Field | Value |
|---|---|
| Universe | `BTCUSDT`, `ETHUSDT` |
| Source | Binance OHLCV |
| Interval | 1 hour |
| Lookback | 730 days |
| OHLCV rows | 17,520 per symbol |
| Feature rows | 17,460 per symbol |
| Target rows | 17,436 per symbol |
| Database | `data/market.duckdb` locally; not committed |

The committed repository keeps curated research artifacts only. Raw data, DuckDB files, model weights, dense row-level predictions, and numpy arrays remain ignored.

## Feature Set

The feature matrix contains 22 engineered technical and microstructure-inspired features:

```text
ret_1h, ret_5h, ret_15h, ret_60h,
vol_5h, vol_20h, vol_of_vol,
amihud, volume_zscore, ret_autocorr,
spread_proxy, ofi_proxy, rsi_14, gk_vol,
skewness, kurtosis, macd_signal,
bband_pct_b, atr_14, close_vs_vwap,
log_vol_trend, ret_dispersion
```

## Target Labels

| Field | Value |
|---|---|
| Primary target | `tb_label_8h` |
| Classes | `-1`, `0`, `+1` |
| Return column | `forward_return_8h` |
| Additional horizons | 4h, 8h, 24h |
| Labeling style | Directional, triple-barrier, forward return, volatility-adjusted return |

The primary alpha benchmark uses the 8-hour triple-barrier target. Phase 14A also evaluates 4-hour and 24-hour targets for robustness.

## Encoder Architecture

| Field | Value |
|---|---|
| Encoder class | `TemporalEncoder` |
| Input shape | `(batch, 60, 22)` |
| Window size | 60 bars |
| Input projection | 22 to 64 |
| Positional encoding | Learnable, max length 512 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Feed-forward dimension | 256 |
| Dropout | 0.1 |
| Pooling | Mean pooling over time |
| Projection head | Linear 64 to 64, ReLU, Linear 64 to 16 |
| Embedding dimension | 16 |
| Output normalization | L2 normalization |
| Parameter count | 139,408 |

## Encoder Training

| Field | Value |
|---|---|
| Current loss | NT-Xent |
| Temperature | 0.07 |
| Positive pair | Adjacent/augmented temporal window from `WindowDataset` |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Scheduler | CosineAnnealingLR |
| Default epochs | 30 |
| Sparse artifact stride | 4 |
| Dense inference stride | 1 in `visualize_regimes.py` |
| Random state | 42 where supported |

This encoder is the main candidate for future upgrades. The current research finding is that vanilla contrastive embeddings improve when HMM state dynamics are added, but they do not yet dominate raw-feature HMM regimes.

## Regime Methods

| Method | Description |
|---|---|
| `contrastive` | GMM over contrastive encoder embeddings |
| `contrastive_hmm` | Gaussian HMM over contrastive encoder embeddings |
| `hmm` | Gaussian HMM over raw selected features |
| `kmeans` | KMeans over raw feature space |
| `vol_bucket` | Realized-volatility quantile buckets |

HMM states are not ground-truth regimes. They are a classical reference proxy and a competitive baseline. Agreement with HMM states should be described as proxy agreement, not accuracy.

## Alpha Models

| Field | Value |
|---|---|
| Model family | LightGBM multiclass classifier |
| Classes | `-1`, `0`, `+1` mapped to 3 classes |
| Alpha score | `P(+1) - P(-1)` |
| Default trade threshold | 0.05 |
| Default transaction cost | 10 bps |
| Global model | One LightGBM over all rows |
| Regime model | One LightGBM per regime, posterior-weighted predictions |

LightGBM parameters:

```text
objective=multiclass
num_class=3
n_estimators=250
learning_rate=0.04
max_depth=4
num_leaves=15
min_child_samples=25
subsample=0.8
colsample_bytree=0.8
random_state=42
```

## Validation

| Field | Value |
|---|---|
| Validation style | Expanding walk-forward |
| Initial training window | 6 months, 4,320 hourly bars |
| Test step | 1 month, 720 hourly bars |
| Embargo | 5 days, 120 hourly bars |
| Primary label horizon purge | 8 bars |
| Main OOS rows | 25,920 per method |
| Critical audit status | 20 PASS, 1 methodological WARN, 0 FAIL |

The current methodological warning is that the legacy `regime_assignments.csv` artifact is offline/global. Predictive regime claims should use the fold-local Phase 13 artifacts.

## Robustness Coverage

| Phase | Coverage |
|---|---|
| Phase 14A | BTC-only, ETH-only, BTC+ETH across 4h, 8h, 24h |
| Phase 14B | Thresholds 0.03, 0.05, 0.07, 0.10; costs 5, 10, 20 bps; all/bull/sideways/bear periods |

Phase 14B re-scores existing fold-local predictions. It does not retrain models for every cost or threshold setting.

## Current Baseline Finding

The current frozen baseline supports a cautious conclusion:

```text
Raw-feature HMM is the strongest current regime-aware layer for signal IC,
Sharpe, and total-return robustness. The global model is more defensive on
drawdown under higher costs. Contrastive-HMM improves the learned-regime path
but does not yet dominate raw-feature HMM.
```

Future encoder phases must be evaluated against this frozen run, not against overwritten latest CSVs.
