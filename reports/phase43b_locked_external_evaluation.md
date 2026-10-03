# Phase 43B Locked External Holdout Evaluation

## Run Status

This is a **locked confirmatory evaluation** over 18 fold(s) and 10 symbol(s), using at most 5,000 deterministic windows per encoder stage. This is the frozen locked-holdout evaluation; the result must be reported once without same-holdout retuning.

## Validity Contract

- Scalers, weak-supervision HMMs, contrastive pairs, encoder weights, assignment layers, and alpha models are fit inside each outer fold.
- Epochs are selected on an inner chronological validation block.
- The selected epoch count is refit on the full authorized outer-training interval.
- Outer-test metrics do not influence training or model selection.
- Vanilla and guided learned methods share the same outer folds and test coverage as classical and global baselines.

## Method Results

| method | IC | Sharpe | drawdown | total_return | turnover | n_test_rows |
| --- | --- | --- | --- | --- | --- | --- |
| global_lgbm | -0.0042 | -1.2810 | -0.1674 | -0.1638 | 0.0470 | 129600 |
| regime_lgbm_contrastive | -0.0002 | 0.2726 | -0.0990 | 0.0355 | 0.0713 | 129600 |
| regime_lgbm_contrastive_hmm | -0.0012 | -0.7277 | -0.1537 | -0.1088 | 0.0765 | 129600 |
| regime_lgbm_hmm | -0.0024 | -0.9538 | -0.1799 | -0.1600 | 0.0736 | 129600 |
| regime_lgbm_hmm_guided_gmm | 0.0072 | -1.7041 | -0.2943 | -0.2718 | 0.0845 | 129600 |
| regime_lgbm_hmm_guided_hmm | 0.0007 | -0.3691 | -0.1144 | -0.0659 | 0.0860 | 129600 |
| regime_lgbm_kmeans | -0.0020 | -0.3003 | -0.1378 | -0.0552 | 0.0761 | 129600 |
| regime_lgbm_vol_bucket | -0.0017 | -0.6418 | -0.1833 | -0.1113 | 0.0847 | 129600 |

## Selected Epochs

| encoder_method | min | median | max |
| --- | --- | --- | --- |
| guided | 1 | 3.0000 | 6 |
| vanilla | 6 | 14.0000 | 30 |

## Interpretation Rule

This locked run evaluates the Phase 43A frozen candidate and references on the Phase 43B registered external holdout. The result may confirm or fail to confirm the guided-HMM mechanism, but it must not be used to tune thresholds, candidates, features, labels, or architecture.
