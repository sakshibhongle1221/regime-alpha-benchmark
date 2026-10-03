# Phase 43A Locked Holdout Freeze

## Status

Phase 43A freezes the final confirmatory protocol before any locked-holdout outcome is inspected.

- Freeze ID: `phase43-locked-holdout-freeze-v1`
- Data role: `locked_unobserved_until_phase43b`
- Final candidate: `regime_lgbm_hmm_guided_hmm`
- Support artifacts hashed: 8

No locked/final data is evaluated in Phase 43A.

## Frozen Candidate And References

| Type | Name | Status | Role |
|---|---|---|---|
| `final_candidate` | `regime_lgbm_hmm_guided_hmm` | frozen | primary locked-holdout candidate |
| `primary_reference` | `global_lgbm` | frozen | primary comparison |
| `primary_reference` | `regime_lgbm_hmm` | frozen | primary comparison |
| `secondary_diagnostic_reference` | `regime_lgbm_kmeans` | frozen | secondary diagnostic comparison |
| `secondary_diagnostic_reference` | `regime_lgbm_vol_bucket` | frozen | secondary diagnostic comparison |
| `secondary_diagnostic_reference` | `regime_lgbm_hmm_guided_gmm` | frozen | secondary diagnostic comparison |
| `excluded_rescue_path` | `phase41b_probability_calibration` | excluded | forbidden before locked evaluation |
| `excluded_rescue_path` | `phase41b_soft_gating` | excluded | forbidden before locked evaluation |
| `excluded_rescue_path` | `score_threshold_execution_control` | excluded | forbidden before locked evaluation |
| `excluded_rescue_path` | `new_architecture_search` | excluded | forbidden before locked evaluation |
| `excluded_rescue_path` | `new_feature_selection` | excluded | forbidden before locked evaluation |
| `excluded_rescue_path` | `new_label_or_horizon_selection` | excluded | forbidden before locked evaluation |

## Locked Holdout Rules

| Rule | Status | Value |
|---|---|---|
| `preferred_holdout` | locked_unobserved | external_asset_holdout |
| `source_universe` | prewritten_selection_source | configs/crypto_universe_candidates.csv |
| `selection_rule` | mandatory | Select the next pre-ranked quality-eligible USDT spot assets not included in asset_universe_crypto20.csv after ingestion and quality checks, without inspecting model outcomes. |
| `minimum_assets` | mandatory | 10 |
| `fallback_holdout` | fallback_only | future_temporal_holdout_strictly_after_crypto20_development_endpoint |
| `fallback_rule` | fallback_only | Use only data strictly after 2026-06-15 02:30 Asia/Kolkata, collected and hashed before any model outcome inspection. |
| `minimum_hourly_bars` | quality_gate | 12000 |
| `maximum_gap_hours` | quality_gate | 6 |
| `stable_or_synthetic_assets_forbidden` | quality_gate | True |
| `coverage_and_hash_manifest_required` | quality_gate | True |
| `validation` | locked_protocol | same repaired common-calendar purged walk-forward protocol |
| `target` | locked_protocol | tb_label_8h |
| `horizon_hours` | locked_protocol | 8 |
| `transaction_cost_bps` | locked_protocol | 10 |
| `candidate_selection_on_holdout` | forbidden | forbidden |
| `threshold_selection_on_holdout` | forbidden | forbidden |
| `rerun_after_failure` | forbidden | forbidden |

## Claim Rules

| Rule | Status | Text |
|---|---|---|
| `success_condition` | required | The final candidate must improve over both primary references on mean asset IC and show non-worse transaction-cost-adjusted Sharpe without violating coverage or statistical guardrails. |
| `failure_condition` | required | If the final candidate does not satisfy the success condition, the paper reports a failed locked confirmation without same-holdout tuning. |
| `allowed_claim_if_success` | allowed | The guided-HMM mechanism shows confirmatory support on the registered locked holdout. |
| `allowed_claim_if_failure` | allowed | The guided-HMM mechanism remains a structurally motivated development finding but does not receive locked holdout confirmation. |
| `forbidden_claim_1` | forbidden | tradable strategy |
| `forbidden_claim_2` | forbidden | guaranteed alpha |
| `forbidden_claim_3` | forbidden | final-test tuning improved performance |
| `forbidden_claim_4` | forbidden | Phase 41B calibration rescued the model |
| `forbidden_claim_5` | forbidden | threshold tuning rescued the model |

## Paper-Safe Interpretation

Phase 43A does not improve the model and does not produce a performance result. It prevents future result-chasing by freezing exactly one final mechanism path before the locked evaluation.

Allowed wording:

```text
We froze the guided-HMM mechanism before locked holdout evaluation; the next result will be reported whether positive or negative.
```

Forbidden wording:

```text
Phase 43A proves the model generalizes.
Phase 43A selects a winner from locked data.
Phase 43A permits threshold tuning after holdout inspection.
```
