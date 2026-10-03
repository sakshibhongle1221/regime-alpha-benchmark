from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase48_paper_polish as phase48  # noqa: E402
import phase49_submission_closure as phase49  # noqa: E402


class Phase49SubmissionClosureTests(unittest.TestCase):
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

    def test_phase49_tex_adds_figures_and_citations(self) -> None:
        tex = phase49.build_phase49_tex(self.locked_results(), self.locked_primary(), self.evidence_matrix())

        self.assertIn("figures/phase49_f1_validation_repair.pdf", tex)
        self.assertIn("figures/phase49_f2_system_overview.pdf", tex)
        self.assertIn("ts2vec2022", tex)
        self.assertIn("tfc2022", tex)
        self.assertIn("cannot replace the frozen candidate", tex)
        self.assertIn("cannot be reused for model rescue", tex)

    def test_phase49_bib_contains_time_series_references(self) -> None:
        bib = phase49.build_phase49_bib()

        for key in phase49.TIME_SERIES_REFERENCES:
            self.assertIn("{" + key + ",", bib)

    def test_citation_audit_requires_tex_and_bib_coverage(self) -> None:
        tex = phase49.build_phase49_tex(self.locked_results(), self.locked_primary(), self.evidence_matrix())
        bib = phase49.build_phase49_bib()
        audit = phase49.build_citation_audit(tex, bib)

        self.assertEqual(set(audit["status"]), {"pass"})

    def test_metadata_audit_flags_identity_leaks(self) -> None:
        metadata_audit = pd.DataFrame(
            [
                {"check_id": "metadata_author", "status": "fail", "detail": "Sakshi Bhongle"},
            ]
        )
        figure_manifest = phase49.build_figure_manifest()
        citation_audit = phase49.build_citation_audit(
            phase49.build_phase49_tex(self.locked_results(), self.locked_primary(), self.evidence_matrix()),
            phase49.build_phase49_bib(),
        )
        checklist = phase49.build_artifact_release_checklist()
        warning_audit = phase49.build_latex_warning_audit(
            {
                "pdf_exists": True,
                "pdf_bytes": 100000,
                "pdf_pages": 3,
                "latex_log_exists": True,
                "bibtex_log_exists": True,
                "warning_markers": [],
            }
        )
        submission = phase49.build_submission_audit(figure_manifest, citation_audit, metadata_audit, checklist, warning_audit)
        status_map = submission.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["metadata_anonymity"], "fail")

    def test_artifact_checklist_does_not_claim_doi(self) -> None:
        checklist = phase49.build_artifact_release_checklist()

        self.assertIn("not_claimed", set(checklist["status"]))

    def test_latex_warning_audit_records_review_required_warnings(self) -> None:
        audit = phase49.build_latex_warning_audit(
            {
                "pdf_exists": True,
                "pdf_bytes": 200000,
                "pdf_pages": 3,
                "latex_log_exists": True,
                "bibtex_log_exists": True,
                "warning_markers": ["LaTeX Warning"],
            }
        )
        status_map = audit.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["pdf_compilation"], "pass")
        self.assertEqual(status_map["warning_cleanup"], "review_required")


if __name__ == "__main__":
    unittest.main()
