from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase46_final_research_completion as phase46  # noqa: E402


class Phase46FinalResearchCompletionTests(unittest.TestCase):
    def locked_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "method": phase46.FINAL_CANDIDATE,
                    "Sharpe": -0.3691,
                    "total_return": -0.0659,
                    "mean_asset_IC": 0.0007,
                },
                {
                    "method": "global_lgbm",
                    "Sharpe": -1.2810,
                    "total_return": -0.1638,
                    "mean_asset_IC": -0.0042,
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

    def venue_audit(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "requirement_id": "page",
                    "requirement": "eight total pages, self-contained",
                    "phase45_action": "double-blind sigconf public archival repository with a DOI",
                }
            ]
        )

    def research_gate(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": "example", "status": "PASS", "detail": "ok"}])

    def manuscript(self) -> str:
        return (
            "This manuscript does not claim a tradable strategy. "
            "The same locked holdout cannot be reused. "
            "It reports limited locked relative support and mentions a persistent repository."
        )

    def test_validate_inputs_accepts_safe_completion_state(self) -> None:
        phase46.validate_inputs(
            self.manuscript(),
            self.venue_audit(),
            self.locked_results(),
            self.locked_claims(),
            self.research_gate(),
        )

    def test_validate_inputs_rejects_profitable_locked_candidate_assumption(self) -> None:
        results = self.locked_results()
        results.loc[results["method"] == phase46.FINAL_CANDIDATE, "Sharpe"] = 0.5

        with self.assertRaisesRegex(ValueError, "non-profitable"):
            phase46.validate_inputs(
                self.manuscript(),
                self.venue_audit(),
                results,
                self.locked_claims(),
                self.research_gate(),
            )

    def test_claim_audit_blocks_tradability_and_holdout_rescue(self) -> None:
        audit = phase46.build_claim_audit(self.locked_results(), self.locked_claims())
        text = " ".join(audit.astype(str).agg(" ".join, axis=1).tolist())

        self.assertIn("Positive tradable alpha is not supported", text)
        self.assertIn("same locked holdout cannot be reused", text.lower())
        self.assertIn("Claim ACM artifact availability before a DOI", text)

    def test_submission_gate_matrix_marks_external_submission_not_yet(self) -> None:
        gates = phase46.build_submission_gate_matrix()
        gate_map = gates.set_index("gate")["status"].to_dict()

        self.assertEqual(gate_map["final_submission"], "not_yet")
        self.assertEqual(gate_map["artifact_availability"], "not_claimed")
        self.assertEqual(gate_map["locked_holdout_integrity"], "pass")

    def test_acm_skeleton_contains_anonymous_review_sigconf(self) -> None:
        section_budget = phase46.build_section_budget()
        skeleton = phase46.build_acm_skeleton(section_budget)

        self.assertIn(r"\documentclass[sigconf,anonymous,review]{acmart}", skeleton)
        self.assertIn("does not claim a tradable strategy", skeleton)
        self.assertIn("No same-holdout rescue", skeleton)

    def test_anonymity_audit_detects_identity_risk_patterns(self) -> None:
        audit = phase46.build_anonymity_audit("No author block but Sakshi appears in text.")
        status_map = audit.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["personal_name_sakshi"], "review_required")
        self.assertEqual(status_map["author_block"], "pass")

    def test_anonymity_audit_detects_real_author_block(self) -> None:
        audit = phase46.build_anonymity_audit(r"\author{Jane Doe}")
        status_map = audit.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["author_block"], "review_required")


if __name__ == "__main__":
    unittest.main()
