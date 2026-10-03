import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repaired_classical_baseline import EXPECTED_METHODS, calendar_bounds, validate_fold_frames


class RepairedClassicalBaselineTests(unittest.TestCase):
    def test_expected_method_ladder_is_frozen(self):
        self.assertEqual(
            EXPECTED_METHODS,
            {
                "global_lgbm",
                "regime_lgbm_hmm",
                "regime_lgbm_kmeans",
                "regime_lgbm_vol_bucket",
            },
        )

    def test_calendar_bounds_reject_overlap(self):
        frame = pd.DataFrame(
            {
                "feat_idx": [0, 1, 0, 1],
                "open_time": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
                ),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "calendar overlap"):
            calendar_bounds(frame, 1, 1, 2)

    def test_unequal_method_coverage_is_rejected(self):
        prediction_rows = []
        coverage_rows = []
        for method_index, method in enumerate(sorted(EXPECTED_METHODS)):
            count = 2 if method_index == 0 else 1
            for idx in range(count):
                prediction_rows.append(
                    {
                        "method": method,
                        "symbol": "AAA",
                        "feat_idx": idx,
                        "fold": 1,
                    }
                )
            coverage_rows.append({"method": method})
        frames = {
            "predictions": pd.DataFrame(prediction_rows),
            "coverage": pd.DataFrame(coverage_rows),
        }
        with self.assertRaisesRegex(RuntimeError, "unequal prediction coverage"):
            validate_fold_frames(frames, 1)


if __name__ == "__main__":
    unittest.main()
