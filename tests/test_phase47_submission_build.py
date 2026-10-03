from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase47_submission_build as phase47  # noqa: E402
import research_grade_checks as rg_checks  # noqa: E402


class Phase47SubmissionBuildTests(unittest.TestCase):
    def locked_results(self) -> pd.DataFrame:
        rows = []
        for method, ic, sharpe, ret, drawdown in [
            (phase47.FINAL_CANDIDATE, 0.000726, -0.3691, -0.0659, -0.1144),
            ("global_lgbm", -0.004174, -1.2810, -0.1638, -0.1674),
            ("regime_lgbm_hmm", -0.002378, -0.9538, -0.1600, -0.1799),
            ("regime_lgbm_hmm_guided_gmm", 0.007158, -1.7041, -0.2718, -0.2943),
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

    def locked_claims(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"claim_id": "locked_relative_success_rule", "claim_status": "satisfied"},
                {"claim_id": "positive_tradable_alpha", "claim_status": "not_supported"},
                {"claim_id": "same_holdout_retuning", "claim_status": "forbidden"},
            ]
        )

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

    def claim_audit(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "claim_id": "positive_tradable_alpha",
                    "safe_wording": "Positive tradable alpha is not supported.",
                    "blocked_wording": "The strategy is profitable or deployable.",
                }
            ]
        )

    def gate_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"gate": "final_submission", "status": "not_yet"},
                {"gate": "claim_control", "status": "pass"},
            ]
        )

    def research_gate(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": "gate", "status": "PASS", "detail": "ok"}])

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

    def test_validate_inputs_accepts_phase46_safe_state(self) -> None:
        phase47.validate_inputs(
            "does not claim a tradable strategy with limited locked relative support, negative Sharpe, and negative total return",
            r"\documentclass[sigconf,anonymous,review]{acmart}",
            self.locked_results(),
            self.locked_claims(),
            self.locked_primary(),
            self.claim_audit(),
            self.gate_matrix(),
            self.research_gate(),
        )

    def test_validate_inputs_rejects_bad_claim_state(self) -> None:
        claims = self.locked_claims()
        claims.loc[claims["claim_id"] == "positive_tradable_alpha", "claim_status"] = "supported"

        with self.assertRaisesRegex(ValueError, "not_supported"):
            phase47.validate_inputs(
                "does not claim a tradable strategy with limited locked relative support, negative Sharpe, and negative total return",
                r"\documentclass[sigconf,anonymous,review]{acmart}",
                self.locked_results(),
                claims,
                self.locked_primary(),
                self.claim_audit(),
                self.gate_matrix(),
                self.research_gate(),
            )

    def test_submission_tex_preserves_anonymous_acm_and_claim_boundaries(self) -> None:
        tex = phase47.build_submission_tex(
            self.locked_results(),
            self.locked_primary(),
            self.evidence_matrix(),
        )

        self.assertIn(r"\documentclass[sigconf,anonymous,review]{acmart}", tex)
        self.assertIn("does not claim a tradable strategy", tex)
        self.assertIn("locked holdout cannot be reused for model rescue", tex)
        self.assertIn(r"\bibliography{phase47_references}", tex)

    def test_source_anonymity_audit_flags_identity_risk(self) -> None:
        audit = phase47.build_anonymity_audit(r"\author{Jane Doe}", "Sakshi")
        status_map = audit.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["author_block"], "review_required")
        self.assertEqual(status_map["personal_name_sakshi"], "review_required")

    def test_build_audit_marks_pdf_compile_as_conditional(self) -> None:
        tex = phase47.build_submission_tex(
            self.locked_results(),
            self.locked_primary(),
            self.evidence_matrix(),
        )
        refs = phase47.build_references_bib()
        anonymity = phase47.build_anonymity_audit(tex, refs)
        audit = phase47.build_build_audit(tex, refs, anonymity)
        status_map = audit.set_index("check_id")["status"].to_dict()

        self.assertEqual(status_map["pdf_compilation"], "conditional_pass")
        self.assertEqual(status_map["pdf_page_budget"], "conditional_pass")
        self.assertEqual(status_map["pdf_warning_review"], "conditional_pass")
        self.assertEqual(status_map["artifact_archive"], "not_claimed")
        self.assertEqual(status_map["no_profitable_strategy_claim"], "pass")

    def test_build_audit_records_compiled_pdf_evidence_conservatively(self) -> None:
        tex = phase47.build_submission_tex(
            self.locked_results(),
            self.locked_primary(),
            self.evidence_matrix(),
        )
        refs = phase47.build_references_bib()
        anonymity = phase47.build_anonymity_audit(tex, refs)
        audit = phase47.build_build_audit(
            tex,
            refs,
            anonymity,
            {
                "exists": True,
                "pages": 2,
                "bytes": 363597,
                "warnings": ["Overfull \\hbox"],
                "log_available": True,
                "bibtex_log_available": True,
            },
        )
        status_map = audit.set_index("check_id")["status"].to_dict()
        detail_map = audit.set_index("check_id")["detail"].to_dict()

        self.assertEqual(status_map["pdf_compilation"], "pass")
        self.assertEqual(status_map["pdf_page_budget"], "pass")
        self.assertEqual(status_map["pdf_warning_review"], "review_required")
        self.assertIn("pages=2", detail_map["pdf_compilation"])

    def test_build_audit_keeps_pdf_warning_review_required_without_logs(self) -> None:
        tex = phase47.build_submission_tex(
            self.locked_results(),
            self.locked_primary(),
            self.evidence_matrix(),
        )
        refs = phase47.build_references_bib()
        anonymity = phase47.build_anonymity_audit(tex, refs)
        audit = phase47.build_build_audit(
            tex,
            refs,
            anonymity,
            {
                "exists": True,
                "pages": 2,
                "bytes": 363597,
                "warnings": [],
                "log_available": False,
                "bibtex_log_available": False,
            },
        )
        status_map = audit.set_index("check_id")["status"].to_dict()
        detail_map = audit.set_index("check_id")["detail"].to_dict()

        self.assertEqual(status_map["pdf_compilation"], "pass")
        self.assertEqual(status_map["pdf_warning_review"], "review_required")
        self.assertIn("logs are unavailable", detail_map["pdf_warning_review"])

    def test_read_pdf_build_status_detects_bibtex_warnings(self) -> None:
        tmp_pdf = Path("tmp_phase47_warning_test.pdf")
        tmp_log = Path("tmp_phase47_warning_test.log")
        tmp_blg = Path("tmp_phase47_warning_test.blg")
        try:
            tmp_pdf.write_bytes(
                b"%PDF-1.4\n1 0 obj << /Type /Pages >> endobj\n"
                b"2 0 obj << /Type /Page >> endobj\n"
            )
            tmp_log.write_text(
                "Output written on tmp_phase47_warning_test.pdf (1 page, 123 bytes)\n"
                "Overfull \\hbox\n",
                encoding="utf-8",
            )
            tmp_blg.write_text("Warning--empty publisher in example\n", encoding="utf-8")

            status = phase47.read_pdf_build_status(tmp_pdf, tmp_log, tmp_blg)

            self.assertTrue(status["exists"])
            self.assertEqual(status["pages"], 1)
            self.assertIn("Overfull \\hbox", status["warnings"])
            self.assertIn("BibTeX Warning", status["warnings"])
            self.assertTrue(status["log_available"])
            self.assertTrue(status["bibtex_log_available"])
        finally:
            for path in [tmp_pdf, tmp_log, tmp_blg]:
                if path.exists():
                    path.unlink()

    def test_estimate_pdf_pages_counts_page_objects_without_external_dependency(self) -> None:
        tmp_pdf = Path("tmp_phase47_page_count_test.pdf")
        try:
            tmp_pdf.write_bytes(
                b"%PDF-1.4\n1 0 obj << /Type /Pages >> endobj\n"
                b"2 0 obj << /Type /Page >> endobj\n"
                b"3 0 obj << /Type /Page >> endobj\n"
            )

            self.assertEqual(phase47.estimate_pdf_pages(tmp_pdf), 2)
        finally:
            if tmp_pdf.exists():
                tmp_pdf.unlink()

    def test_figure_manifest_uses_phase45_section_and_purpose(self) -> None:
        plan = pd.DataFrame(
            [
                {
                    "figure_id": "F1",
                    "paper_section": "Validation protocol",
                    "artifact": "reports/example.md",
                    "purpose": "Show the repaired fold design.",
                    "status": "needs_final_drawing",
                }
            ]
        )
        manifest = phase47.build_figure_manifest(plan)

        self.assertEqual(manifest.iloc[0]["figure_id"], "F1")
        self.assertIn("Validation protocol", manifest.iloc[0]["title"])
        self.assertIn("repaired fold design", manifest.iloc[0]["title"])
        self.assertEqual(manifest.iloc[0]["phase47_status"], "needs_final_drawing")

    def test_research_gate_checks_phase47_curated_csv_unignore_entries(self) -> None:
        results: list[rg_checks.CheckResult] = []
        rg_checks.check_phase47_submission_build(results)
        result_map = {result.check: result for result in results}

        self.assertIn("phase47_gitignore_curated_csv_guardrails", result_map)
        self.assertEqual(
            result_map["phase47_gitignore_curated_csv_guardrails"].status,
            rg_checks.PASS,
        )


if __name__ == "__main__":
    unittest.main()
