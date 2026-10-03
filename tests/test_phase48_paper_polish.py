from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase48_paper_polish as phase48  # noqa: E402


class Phase48PaperPolishTests(unittest.TestCase):
    def locked_results(self) -> pd.DataFrame:
        rows = []
        for method, ic, sharpe, ret, drawdown in [
            (phase48.FINAL_CANDIDATE, 0.000726, -0.3691, -0.0659, -0.1144),
            ("global_lgbm", -0.004174, -1.2810, -0.1638, -0.1674),
            ("regime_lgbm_hmm", -0.002378, -0.9538, -0.1600, -0.1799),
            ("regime_lgbm_hmm_guided_gmm", 0.007158, -1.7041, -0.2718, -0.2943),
            ("regime_lgbm_contrastive", -0.000230, 0.2726, 0.0355, -0.0990),
        ]:
            rows.append(
                {
                    "method": method,
                    "mean_asset_IC": ic,
                    "Sharpe": sharpe,
                    "total_return": ret,
                    "drawdown": drawdown,
                    "n_test_rows": 129600,
                }
            )
        return pd.DataFrame(rows)

    def locked_primary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "reference_method": "global_lgbm",
                    "delta_mean_asset_IC": 0.0049,
                    "delta_Sharpe": 0.9119,
                    "coverage_equal": True,
                },
                {
                    "reference_method": "regime_lgbm_hmm",
                    "delta_mean_asset_IC": 0.0031,
                    "delta_Sharpe": 0.5847,
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

    def evidence_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "evidence_block": "locked_external_holdout",
                    "data_role": "locked_registered_unobserved",
                    "claim_boundary": "No tradable-alpha claim.",
                }
            ]
        )

    def phase47_audit(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"check_id": "pdf_compilation", "status": "pass"},
                {"check_id": "pdf_warning_review", "status": "review_required"},
            ]
        )

    def research_gate(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": "gate", "status": "PASS", "detail": "ok"}])

    def test_validate_inputs_rejects_profitable_alpha_claim(self) -> None:
        claims = self.locked_claims()
        claims.loc[claims["claim_id"] == "positive_tradable_alpha", "claim_status"] = "supported"

        with self.assertRaisesRegex(ValueError, "positive tradable alpha"):
            phase48.validate_inputs(
                self.locked_results(),
                self.locked_primary(),
                claims,
                self.evidence_matrix(),
                self.phase47_audit(),
                self.research_gate(),
            )

    def test_phase48_tex_preserves_limited_claim_and_blocks_switching(self) -> None:
        tex = phase48.build_tex(self.locked_results(), self.locked_primary(), self.evidence_matrix())

        self.assertIn("limited locked relative", tex)
        self.assertIn("does not make that claim", tex)
        self.assertIn("post-hoc model selection", tex)
        self.assertIn("cannot be reused for model rescue", tex)
        self.assertNotIn("profitable trading strategy", tex.lower())

    def test_claim_traceability_blocks_unsafe_claims(self) -> None:
        trace = phase48.build_claim_traceability(self.locked_primary(), self.locked_claims())
        status_map = trace.set_index("claim_id")["status"].to_dict()

        self.assertEqual(status_map["C1"], "allowed")
        self.assertEqual(status_map["C3"], "blocked")
        self.assertEqual(status_map["C4"], "blocked")
        self.assertEqual(status_map["C5"], "blocked")

    def test_warning_audit_is_conservative_without_logs(self) -> None:
        audit = phase48.build_warning_audit(
            {
                "pdf_exists": True,
                "pdf_pages": 5,
                "pdf_bytes": 1000,
                "latex_log_exists": False,
                "bibtex_log_exists": False,
                "warning_markers": [],
            }
        )
        status_map = audit.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["pdf_compilation"], "pass")
        self.assertEqual(status_map["warning_cleanup"], "review_required")
        self.assertEqual(status_map["build_logs_available"], "review_required")

    def test_quality_audit_rejects_forbidden_phrases(self) -> None:
        trace = phase48.build_claim_traceability(self.locked_primary(), self.locked_claims())
        warning_audit = phase48.build_warning_audit(
            {
                "pdf_exists": False,
                "pdf_pages": None,
                "pdf_bytes": None,
                "latex_log_exists": False,
                "bibtex_log_exists": False,
                "warning_markers": [],
            }
        )
        tex = (
            r"\documentclass[sigconf,anonymous,review]{acmart}"
            " cannot be reused for model rescue post-hoc model selection "
            "not a trading claim presents a profitable trading strategy"
        )

        audit = phase48.build_quality_audit(tex, trace, warning_audit)
        status_map = audit.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["claim_boundaries"], "fail")


if __name__ == "__main__":
    unittest.main()
