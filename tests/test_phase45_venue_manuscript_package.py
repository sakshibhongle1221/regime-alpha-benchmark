from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase45_venue_manuscript_package as phase45  # noqa: E402


class Phase45VenueManuscriptPackageTest(unittest.TestCase):
    def locked_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "method": "global_lgbm",
                    "mean_asset_IC": -0.0042,
                    "Sharpe": -1.2810,
                    "total_return": -0.1638,
                    "drawdown": -0.1674,
                    "n_test_rows": 129600,
                },
                {
                    "method": "regime_lgbm_hmm",
                    "mean_asset_IC": -0.0024,
                    "Sharpe": -0.9538,
                    "total_return": -0.1600,
                    "drawdown": -0.1799,
                    "n_test_rows": 129600,
                },
                {
                    "method": "regime_lgbm_hmm_guided_hmm",
                    "mean_asset_IC": 0.0007,
                    "Sharpe": -0.3691,
                    "total_return": -0.0659,
                    "drawdown": -0.1144,
                    "n_test_rows": 129600,
                },
                {
                    "method": "regime_lgbm_hmm_guided_gmm",
                    "mean_asset_IC": 0.0072,
                    "Sharpe": -1.7041,
                    "total_return": -0.2718,
                    "drawdown": -0.2943,
                    "n_test_rows": 129600,
                },
            ]
        )

    def locked_primary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "final_candidate": "regime_lgbm_hmm_guided_hmm",
                    "reference_method": "global_lgbm",
                    "ic_improved": True,
                    "sharpe_non_worse": True,
                    "coverage_equal": True,
                },
                {
                    "final_candidate": "regime_lgbm_hmm_guided_hmm",
                    "reference_method": "regime_lgbm_hmm",
                    "ic_improved": True,
                    "sharpe_non_worse": True,
                    "coverage_equal": True,
                },
            ]
        )

    def locked_claims(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"claim_id": "locked_relative_success_rule", "claim_status": "satisfied"},
                {"claim_id": "positive_tradable_alpha", "claim_status": "not_supported"},
                {"claim_id": "same_holdout_retuning", "claim_status": "forbidden"},
            ]
        )

    def phase44_evidence(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "evidence_block": "validation_repair",
                    "data_role": "development_observed",
                    "finding": "Repaired validation.",
                    "claim_boundary": "Audit history only.",
                },
                {
                    "evidence_block": "locked_external_holdout",
                    "data_role": "locked_registered_unobserved",
                    "finding": "Locked relative support.",
                    "claim_boundary": "No tradable alpha.",
                },
            ]
        )

    def test_validate_inputs_accepts_frozen_safe_locked_evidence(self) -> None:
        phase45.validate_inputs(
            self.locked_results(),
            self.locked_primary(),
            self.locked_claims(),
            self.phase44_evidence(),
        )

    def test_validate_inputs_rejects_same_holdout_retuning_drift(self) -> None:
        claims = self.locked_claims()
        claims.loc[claims["claim_id"] == "same_holdout_retuning", "claim_status"] = "allowed"

        with self.assertRaisesRegex(ValueError, "same-holdout retuning"):
            phase45.validate_inputs(
                self.locked_results(),
                self.locked_primary(),
                claims,
                self.phase44_evidence(),
            )

    def test_generated_manuscript_keeps_claim_boundaries(self) -> None:
        manuscript = phase45.build_manuscript(self.locked_results(), self.phase44_evidence())

        required_phrases = [
            "does not claim a tradable strategy",
            "same locked holdout cannot be reused",
            "frozen final candidate",
            "limited locked relative support",
            "post-hoc selection",
        ]
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, manuscript)

    def test_claim_section_map_blocks_profitability_and_candidate_switching(self) -> None:
        claim_map = phase45.build_claim_section_map()
        text = " ".join(claim_map.astype(str).agg(" ".join, axis=1).tolist())

        self.assertIn("Do not claim a deployable trading strategy", text)
        self.assertIn("Do not tune thresholds, labels, features, architectures", text)
        self.assertIn("Do not claim broad method dominance", text)

    def test_table_and_figure_plans_have_submission_coverage(self) -> None:
        table_text = " ".join(phase45.build_table_plan().astype(str).agg(" ".join, axis=1).tolist())
        figure_text = " ".join(phase45.build_figure_plan().astype(str).agg(" ".join, axis=1).tolist())

        self.assertIn("Locked external holdout", table_text)
        self.assertIn("Reproducibility appendix", table_text)
        self.assertIn("Validation protocol", figure_text)
        self.assertIn("Mechanism boundary", figure_text)

    def test_venue_requirement_audit_tracks_acm_and_icaif_constraints(self) -> None:
        audit = phase45.build_venue_requirement_audit()
        text = " ".join(audit.astype(str).agg(" ".join, axis=1).tolist())

        self.assertIn("eight total pages", text)
        self.assertIn("double-blind", text)
        self.assertIn("sigconf", text)
        self.assertIn("documented, consistent, complete enough for review, and exercisable", text)
        self.assertIn("public archival repository with a DOI", text)

    def test_external_audit_blocks_artifact_availability_overclaim(self) -> None:
        audit = phase45.build_venue_requirement_audit()
        report = phase45.build_external_audit(audit)

        self.assertIn("should not claim artifact availability until a persistent archived release exists", report)
        self.assertIn("does not touch locked/final evaluation data", report)
        self.assertIn("self-contained PDF", report)


if __name__ == "__main__":
    unittest.main()
