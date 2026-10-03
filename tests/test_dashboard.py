import unittest

import pandas as pd

import dashboard


class FakeColumn:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, str, str | None]] = []

    def metric(self, label: str, value: object, delta: object = None) -> None:
        self.metrics.append((label, str(value), None if delta is None else str(delta)))


class FakeStreamlit:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.tables = 0
        self.metrics: list[tuple[str, str, str | None]] = []

    def header(self, text: str) -> None:
        self.messages.append(("header", text))

    def subheader(self, text: str) -> None:
        self.messages.append(("subheader", text))

    def caption(self, text: str) -> None:
        self.messages.append(("caption", text))

    def warning(self, text: str) -> None:
        self.messages.append(("warning", text))

    def info(self, text: str) -> None:
        self.messages.append(("info", text))

    def dataframe(self, _data: pd.DataFrame, width: str = "stretch") -> None:
        self.tables += 1
        self.messages.append(("dataframe_width", width))

    def columns(self, count: int) -> list[FakeColumn]:
        return [FakeColumn() for _ in range(count)]


class DashboardTests(unittest.TestCase):
    def test_format_float_handles_numbers_missing_and_percent(self) -> None:
        self.assertEqual(dashboard.format_float(0.123456, digits=3), "0.123")
        self.assertEqual(dashboard.format_float(-0.065929, percent=True), "-6.6%")
        self.assertEqual(dashboard.format_float("not-a-number"), "missing")

    def test_current_status_renderer_uses_phase43b_and_phase46_artifacts(self) -> None:
        fake = FakeStreamlit()
        claims = pd.DataFrame(
            [
                {
                    "claim_id": "limited_locked_relative_support",
                    "claim_status": "satisfied",
                    "claim": "relative support only",
                    "evidence": "locked rule satisfied",
                }
            ]
        )
        primary = pd.DataFrame(
            [
                {
                    "final_candidate": "regime_lgbm_hmm_guided_hmm",
                    "reference_method": "global_lgbm",
                    "delta_mean_asset_IC": 0.0049,
                    "delta_Sharpe": 0.91,
                    "coverage_equal": True,
                },
                {
                    "final_candidate": "regime_lgbm_hmm_guided_hmm",
                    "reference_method": "regime_lgbm_hmm",
                    "delta_mean_asset_IC": 0.0031,
                    "delta_Sharpe": 0.58,
                    "coverage_equal": True,
                },
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "method": "regime_lgbm_hmm_guided_hmm",
                    "Sharpe": -0.3691,
                    "total_return": -0.0659,
                }
            ]
        )
        gates = pd.DataFrame(
            [
                {"gate": "claim_control", "status": "pass", "evidence_or_action": "safe"},
                {"gate": "double_blind", "status": "conditional_pass", "evidence_or_action": "audit PDF"},
            ]
        )
        claim_audit = pd.DataFrame(
            [{"claim_id": "positive_tradable_alpha", "status": "not_supported"}]
        )
        checks = pd.DataFrame(
            [
                {"check": "gate", "status": "PASS", "detail": "ok"},
                {"check": "claim", "status": "PASS", "detail": "ok"},
            ]
        )

        dashboard.render_current_research_status(
            fake,
            claims,
            primary,
            results,
            gates,
            claim_audit,
            checks,
        )

        rendered_text = "\n".join(text for _kind, text in fake.messages)
        self.assertIn("Current Research Status: Phase 47", rendered_text)
        self.assertIn("Phase 43B/46/47 artifacts", rendered_text)
        self.assertIn("profitable or deployable trading strategy", rendered_text)
        self.assertGreaterEqual(fake.tables, 3)

    def test_streamlit_app_smoke_renders_without_exceptions(self) -> None:
        try:
            from streamlit.testing.v1 import AppTest
        except ModuleNotFoundError as exc:
            self.skipTest(f"Streamlit test harness unavailable: {exc}")

        app = AppTest.from_file("streamlit_app.py")
        app.run(timeout=30)

        exception_values = [str(exception.value) for exception in app.exception]
        self.assertEqual(exception_values, [])


if __name__ == "__main__":
    unittest.main()
