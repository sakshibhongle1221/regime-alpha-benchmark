import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase44_paper_readiness_package import (  # noqa: E402
    FINAL_CANDIDATE,
    build_evidence_matrix,
    build_paper,
    build_reviewer_brief,
    build_risk_register,
    claim_status,
    validate_inputs,
)


def locked_results_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "global_lgbm",
                "mean_asset_IC": -0.004,
                "Sharpe": -1.2,
                "total_return": -0.16,
                "drawdown": -0.17,
                "n_test_rows": 100,
            },
            {
                "method": "regime_lgbm_hmm",
                "mean_asset_IC": -0.002,
                "Sharpe": -0.9,
                "total_return": -0.15,
                "drawdown": -0.18,
                "n_test_rows": 100,
            },
            {
                "method": "regime_lgbm_hmm_guided_gmm",
                "mean_asset_IC": 0.007,
                "Sharpe": -1.7,
                "total_return": -0.27,
                "drawdown": -0.29,
                "n_test_rows": 100,
            },
            {
                "method": FINAL_CANDIDATE,
                "mean_asset_IC": 0.001,
                "Sharpe": -0.3,
                "total_return": -0.06,
                "drawdown": -0.11,
                "n_test_rows": 100,
            },
        ]
    )


def claims_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"claim_id": "locked_relative_success_rule", "claim_status": "satisfied"},
            {"claim_id": "positive_tradable_alpha", "claim_status": "not_supported"},
            {"claim_id": "candidate_switching_after_holdout", "claim_status": "forbidden"},
            {"claim_id": "same_holdout_retuning", "claim_status": "forbidden"},
        ]
    )


class Phase44PaperReadinessPackageTests(unittest.TestCase):
    def test_validate_inputs_requires_locked_relative_success_without_tradable_claim(self):
        primary = pd.DataFrame(
            [
                {
                    "final_candidate": FINAL_CANDIDATE,
                    "reference_method": "global_lgbm",
                    "ic_improved": True,
                    "sharpe_non_worse": True,
                    "coverage_equal": True,
                },
                {
                    "final_candidate": FINAL_CANDIDATE,
                    "reference_method": "regime_lgbm_hmm",
                    "ic_improved": True,
                    "sharpe_non_worse": True,
                    "coverage_equal": True,
                },
            ]
        )
        validate_inputs(locked_results_fixture(), claims_fixture(), primary)
        bad_claims = claims_fixture().copy()
        bad_claims.loc[
            bad_claims["claim_id"] == "positive_tradable_alpha", "claim_status"
        ] = "supported"
        with self.assertRaisesRegex(ValueError, "tradable-alpha"):
            validate_inputs(locked_results_fixture(), bad_claims, primary)

        bad_primary = primary.copy()
        bad_primary.loc[0, "ic_improved"] = False
        with self.assertRaisesRegex(ValueError, "relative rule"):
            validate_inputs(locked_results_fixture(), claims_fixture(), bad_primary)

    def test_risk_register_blocks_candidate_switching_after_locked_holdout(self):
        risk = build_risk_register()
        text = " ".join(risk.astype(str).agg(" ".join, axis=1).tolist())
        self.assertIn("Candidate switching after holdout is forbidden", text)
        self.assertIn("guided-GMM cannot replace guided-HMM", text)

    def test_evidence_matrix_keeps_locked_claim_limited(self):
        development_results = pd.DataFrame(
            [
                {"method": FINAL_CANDIDATE, "mean_asset_IC": -0.01, "Sharpe": -0.7},
                {"method": "global_lgbm", "mean_asset_IC": -0.02, "Sharpe": -1.0},
            ]
        )
        development_stats = pd.DataFrame(
            [
                {
                    "method": FINAL_CANDIDATE,
                    "n_folds": 16,
                    "IC_ci_low": -0.03,
                    "IC_ci_high": 0.01,
                    "Sharpe_ci_low": -3.0,
                    "Sharpe_ci_high": 0.2,
                }
            ]
        )
        stress = pd.DataFrame(
            [
                {
                    "method": FINAL_CANDIDATE,
                    "positive_return_cells": 2,
                    "stress_cells": 16,
                }
            ]
        )
        evidence = build_evidence_matrix(
            locked_results_fixture(),
            claims_fixture(),
            development_results,
            development_stats,
            stress,
        )
        locked = evidence[evidence["evidence_block"] == "locked_external_holdout"].iloc[0]
        self.assertIn("relative rule is satisfied", locked["paper_use"])
        self.assertIn("Tradable-alpha claim is not_supported", locked["claim_boundary"])

    def test_paper_mentions_no_candidate_switching_and_no_tradable_strategy(self):
        development_results = pd.DataFrame(
            [
                {"method": FINAL_CANDIDATE, "mean_asset_IC": -0.01, "Sharpe": -0.7},
                {"method": "global_lgbm", "mean_asset_IC": -0.02, "Sharpe": -1.0},
            ]
        )
        development_stats = pd.DataFrame(
            [
                {
                    "method": FINAL_CANDIDATE,
                    "n_folds": 16,
                    "IC_ci_low": -0.03,
                    "IC_ci_high": 0.01,
                    "Sharpe_ci_low": -3.0,
                    "Sharpe_ci_high": 0.2,
                }
            ]
        )
        stress = pd.DataFrame(
            [
                {
                    "method": FINAL_CANDIDATE,
                    "positive_return_cells": 2,
                    "stress_cells": 16,
                }
            ]
        )
        evidence = build_evidence_matrix(
            locked_results_fixture(),
            claims_fixture(),
            development_results,
            development_stats,
            stress,
        )
        paper = build_paper(locked_results_fixture(), evidence)
        self.assertIn("does not claim a tradable strategy", paper)
        self.assertIn("may not switch to a secondary diagnostic method", paper)
        self.assertEqual(
            claim_status(claims_fixture(), "positive_tradable_alpha"),
            "not_supported",
        )

    def test_reviewer_brief_answers_expected_objections(self):
        brief = build_reviewer_brief(locked_results_fixture())
        self.assertIn("validation-and-mechanism paper", brief)
        self.assertIn("No. The locked candidate has negative Sharpe", brief)
        self.assertIn("Candidate switching after locked evaluation is forbidden", brief)
        self.assertIn("do not reuse the same locked holdout for model rescue", brief)


if __name__ == "__main__":
    unittest.main()
