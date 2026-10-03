import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase43b_locked_holdout_adjudication import (
    FINAL_CANDIDATE,
    build_claims,
    build_primary_comparison,
)


class Phase43BLockedHoldoutAdjudicationTests(unittest.TestCase):
    def test_primary_rule_satisfied_when_candidate_beats_references(self):
        results = pd.DataFrame(
            [
                {"method": "global_lgbm", "mean_asset_IC": -0.01, "Sharpe": -1.0, "total_return": -0.2, "drawdown": -0.2, "n_test_rows": 100},
                {"method": "regime_lgbm_hmm", "mean_asset_IC": -0.02, "Sharpe": -0.5, "total_return": -0.1, "drawdown": -0.1, "n_test_rows": 100},
                {"method": FINAL_CANDIDATE, "mean_asset_IC": 0.001, "Sharpe": -0.4, "total_return": -0.05, "drawdown": -0.1, "n_test_rows": 100},
            ]
        )
        primary = build_primary_comparison(results)
        self.assertTrue(primary["ic_improved"].all())
        self.assertTrue(primary["sharpe_non_worse"].all())
        self.assertTrue(primary["coverage_equal"].all())

    def test_claims_do_not_convert_relative_success_into_tradable_claim(self):
        results = pd.DataFrame(
            [
                {"method": "global_lgbm", "mean_asset_IC": -0.01, "Sharpe": -1.0, "total_return": -0.2, "drawdown": -0.2, "n_test_rows": 100},
                {"method": "regime_lgbm_hmm", "mean_asset_IC": -0.02, "Sharpe": -0.5, "total_return": -0.1, "drawdown": -0.1, "n_test_rows": 100},
                {"method": FINAL_CANDIDATE, "mean_asset_IC": 0.001, "Sharpe": -0.4, "total_return": -0.05, "drawdown": -0.1, "n_test_rows": 100},
            ]
        )
        primary = build_primary_comparison(results)
        fold_metrics = pd.DataFrame(
            [{"method": method, "fold": fold} for method in results["method"] for fold in [1, 2]]
        )
        claims = build_claims(primary, results, fold_metrics, {"fold_count": 2}).set_index("claim_id")
        self.assertEqual(claims.loc["locked_relative_success_rule", "claim_status"], "satisfied")
        self.assertEqual(claims.loc["positive_tradable_alpha", "claim_status"], "not_supported")


if __name__ == "__main__":
    unittest.main()
