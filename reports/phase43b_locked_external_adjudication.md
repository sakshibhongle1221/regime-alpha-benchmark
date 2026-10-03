# Phase 43B Locked External Holdout Adjudication

## Status

The Phase 43B locked external evaluation is complete and adjudicated under the Phase 43A frozen rule.

- Freeze ID: `phase43b-locked-external-holdout-v1`
- Registration status: `registered_ready`
- Registered symbols: `FILUSDT,ARBUSDT,OPUSDT,INJUSDT,ATOMUSDT,XLMUSDT,ALGOUSDT,AAVEUSDT,RUNEUSDT,FETUSDT`
- Fold count: 18
- Prediction rows: 173770
- Final candidate: `regime_lgbm_hmm_guided_hmm`
- Primary locked claim status: `satisfied`

## Locked Result Summary

| method | mean_asset_IC | Sharpe | total_return | drawdown | n_test_rows |
| --- | --- | --- | --- | --- | --- |
| global_lgbm | -0.0042 | -1.2810 | -0.1638 | -0.1674 | 129600 |
| regime_lgbm_contrastive | -0.0002 | 0.2726 | 0.0355 | -0.0990 | 129600 |
| regime_lgbm_contrastive_hmm | -0.0012 | -0.7277 | -0.1088 | -0.1537 | 129600 |
| regime_lgbm_hmm | -0.0024 | -0.9538 | -0.1600 | -0.1799 | 129600 |
| regime_lgbm_hmm_guided_gmm | 0.0072 | -1.7041 | -0.2718 | -0.2943 | 129600 |
| regime_lgbm_hmm_guided_hmm | 0.0007 | -0.3691 | -0.0659 | -0.1144 | 129600 |
| regime_lgbm_kmeans | -0.0020 | -0.3003 | -0.0552 | -0.1378 | 129600 |
| regime_lgbm_vol_bucket | -0.0017 | -0.6418 | -0.1113 | -0.1833 | 129600 |

## Primary Frozen Rule Comparison

| final_candidate | reference_method | candidate_mean_asset_IC | reference_mean_asset_IC | delta_mean_asset_IC | candidate_Sharpe | reference_Sharpe | delta_Sharpe | candidate_total_return | reference_total_return | candidate_drawdown | reference_drawdown | candidate_n_test_rows | reference_n_test_rows | ic_improved | sharpe_non_worse | coverage_equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regime_lgbm_hmm_guided_hmm | global_lgbm | 0.0007 | -0.0042 | 0.0049 | -0.3691 | -1.2810 | 0.9119 | -0.0659 | -0.1638 | -0.1144 | -0.1674 | 129600 | 129600 | True | True | True |
| regime_lgbm_hmm_guided_hmm | regime_lgbm_hmm | 0.0007 | -0.0024 | 0.0031 | -0.3691 | -0.9538 | 0.5847 | -0.0659 | -0.1600 | -0.1144 | -0.1799 | 129600 | 129600 | True | True | True |

## Claim Adjudication

| claim_id | claim_status | claim | evidence |
| --- | --- | --- | --- |
| locked_relative_success_rule | satisfied | The frozen guided-HMM final candidate improves over both primary references on mean asset IC and has non-worse transaction-cost-adjusted Sharpe with equal coverage. | All primary comparisons satisfy IC/Sharpe/coverage requirements. |
| positive_tradable_alpha | not_supported | The locked result supports a tradable positive-alpha strategy. | Final candidate Sharpe=-0.3691, total_return=-0.0659; positive_alpha=False. |
| candidate_switching_after_holdout | forbidden | A non-frozen method can replace the final candidate after locked outcome inspection. | A secondary diagnostic method has higher locked IC, but it was not the frozen final candidate and cannot replace the final candidate after outcome inspection. |
| same_holdout_retuning | forbidden | Thresholds, features, labels, architecture, or method choice can be tuned after locked evaluation. | Phase 43A and Phase 43B forbid same-holdout tuning or rerun-after-failure. |

## Paper-Safe Interpretation

The frozen guided-HMM candidate satisfies the prewritten relative locked-holdout rule against `global_lgbm` and `regime_lgbm_hmm`: mean asset IC is higher and Sharpe is non-worse with equal coverage.

This is **not** a tradable-strategy claim. The final candidate still has negative locked Sharpe (-0.3691) and negative locked total return (-0.0659).

A secondary or diagnostic method cannot replace the final candidate after locked evaluation.

The paper-safe wording is:

```text
The frozen guided-HMM mechanism receives limited locked-holdout support on the pre-specified relative IC/Sharpe rule, but it does not establish a profitable or tradable strategy.
```

Forbidden wording:

```text
The locked holdout proves a tradable strategy.
We can switch the final method after seeing the locked holdout.
The locked holdout result authorizes threshold or model retuning.
```
