# Phase 39 Fully Fold-Local Encoder Results

## Run Status

This is a **full development-observed benchmark** over 16 fold(s) and 20 symbol(s), using at most 5,000 deterministic windows per encoder stage. All outcomes remain development-observed and cannot be used as an untouched final test.

## Validity Contract

- Scalers, weak-supervision HMMs, contrastive pairs, encoder weights, assignment layers, and alpha models are fit inside each outer fold.
- Epochs are selected on an inner chronological validation block.
- The selected epoch count is refit on the full authorized outer-training interval.
- Outer-test metrics do not influence training or model selection.
- Vanilla and guided learned methods share the same outer folds and test coverage as classical and global baselines.

## Method Results

| method | IC | Sharpe | drawdown | total_return | turnover | n_test_rows |
| --- | --- | --- | --- | --- | --- | --- |
| global_lgbm | -0.0113 | -0.0884 | -0.0619 | -0.0089 | 0.0395 | 230400 |
| regime_lgbm_contrastive | -0.0031 | -0.7965 | -0.0797 | -0.0597 | 0.0539 | 230400 |
| regime_lgbm_contrastive_hmm | -0.0178 | -1.1657 | -0.1175 | -0.0851 | 0.0596 | 230400 |
| regime_lgbm_hmm | -0.0093 | -0.2600 | -0.0843 | -0.0258 | 0.0616 | 230400 |
| regime_lgbm_hmm_guided_gmm | -0.0087 | -1.0826 | -0.0977 | -0.0895 | 0.0666 | 230400 |
| regime_lgbm_hmm_guided_hmm | -0.0119 | -0.7620 | -0.1018 | -0.0621 | 0.0674 | 230400 |
| regime_lgbm_kmeans | -0.0139 | -0.2091 | -0.0464 | -0.0188 | 0.0588 | 230400 |
| regime_lgbm_vol_bucket | -0.0153 | -0.2954 | -0.0674 | -0.0267 | 0.0711 | 230400 |

## Selected Epochs

| encoder_method | min | median | max |
| --- | --- | --- | --- |
| guided | 1 | 4.0000 | 22 |
| vanilla | 4 | 14.5000 | 24 |

## Interpretation Rule

A smoke run validates code paths, leakage boundaries, artifacts, and coverage only. A full run is still development evidence. Model changes require a new registered candidate family, and confirmatory claims require a frozen configuration evaluated once on a locked holdout.
