import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase42_interpretation_execution import (
    apply_score_threshold,
    execution_summary,
    stress_grid,
    transition_diagnostics,
)


class Phase42InterpretationExecutionTests(unittest.TestCase):
    def test_apply_score_threshold_uses_score_and_neutral_best_class(self):
        frame = pd.DataFrame(
            {
                "prob_down": [0.1, 0.6, 0.1],
                "prob_neutral": [0.2, 0.3, 0.8],
                "prob_up": [0.7, 0.1, 0.1],
                "score": [0.6, -0.5, 0.0],
                "signal": [0, 0, 0],
            }
        )
        out = apply_score_threshold(frame, threshold=0.25)
        self.assertEqual(out["signal"].tolist(), [1, -1, 0])

    def test_stress_grid_and_summary_keep_benchmark_method_scope(self):
        frame = pd.DataFrame(
            {
                "benchmark": ["b1"] * 4,
                "method": ["m1"] * 4,
                "symbol": ["BTCUSDT"] * 4,
                "open_time": pd.date_range("2026-01-01", periods=4, freq="8h", tz="UTC"),
                "target_return": [0.01, -0.01, 0.02, -0.02],
                "target_label": [1, -1, 1, -1],
                "fold": [1, 1, 1, 1],
                "prob_down": [0.1, 0.7, 0.1, 0.7],
                "prob_neutral": [0.2, 0.2, 0.2, 0.2],
                "prob_up": [0.7, 0.1, 0.7, 0.1],
                "score": [0.6, -0.6, 0.6, -0.6],
                "signal": [1, -1, 1, -1],
            }
        )
        stress = stress_grid(frame, thresholds=[0.0], cost_bps_values=[10.0])
        summary = execution_summary(stress)
        self.assertEqual(len(stress), 1)
        self.assertEqual(summary.loc[0, "benchmark"], "b1")
        self.assertEqual(summary.loc[0, "method"], "m1")
        self.assertEqual(summary.loc[0, "stress_cells"], 1)

    def test_transition_diagnostics_detects_switch_rate(self):
        assignments = pd.DataFrame(
            {
                "method": ["hmm"] * 4,
                "fold": [1, 1, 1, 1],
                "symbol": ["BTCUSDT"] * 4,
                "feat_idx": [1, 2, 3, 4],
                "regime": [0, 0, 1, 1],
                "post_0": [0.9, 0.8, 0.2, 0.1],
                "post_1": [0.1, 0.2, 0.8, 0.9],
            }
        )
        out = transition_diagnostics(assignments)
        self.assertEqual(out.loc[0, "regime_method"], "hmm")
        self.assertAlmostEqual(out.loc[0, "switch_rate"], 0.25)
        self.assertAlmostEqual(out.loc[0, "avg_duration"], 2.0)


if __name__ == "__main__":
    unittest.main()
