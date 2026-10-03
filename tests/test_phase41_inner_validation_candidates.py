import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase41_inner_validation_candidates import (
    PHASE41B_DEFERRED_CANDIDATES,
    PHASE41B_EXECUTED_CANDIDATES,
    candidate_frames,
    split_inner_validation,
)


class Phase41InnerValidationCandidateTests(unittest.TestCase):
    def test_inner_validation_split_respects_gap(self):
        rows = []
        for i, ts in enumerate(pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")):
            rows.append({"row_id": i, "open_time": ts, "symbol": "BTCUSDT"})
        frame = pd.DataFrame(rows).set_index("row_id", drop=False)
        inner_train, inner_val = split_inner_validation(
            frame,
            list(range(200)),
            validation_bars=24,
            embargo_bars=10,
            purge_bars=2,
        )
        train_end = frame.loc[inner_train, "open_time"].max()
        val_start = frame.loc[inner_val, "open_time"].min()
        self.assertGreaterEqual(val_start - train_end, pd.Timedelta(hours=12))
        self.assertEqual(len(inner_val), 24)

    def test_global_shrinkage_aligns_by_row_id(self):
        base = pd.DataFrame(
            {
                "row_id": [2, 1],
                "prob_down": [0.8, 0.1],
                "prob_neutral": [0.1, 0.1],
                "prob_up": [0.1, 0.8],
                "score": [-0.7, 0.7],
                "pred_label": [-1, 1],
                "signal": [-1, 1],
            }
        )
        global_reference = pd.DataFrame(
            {
                "row_id": [1, 2],
                "prob_down": [0.0, 1.0],
                "prob_neutral": [0.0, 0.0],
                "prob_up": [1.0, 0.0],
            }
        )
        config = {
            "candidate_grids": {
                "probability_temperature": [],
                "prior_blend_weight": [],
                "global_regime_shrinkage": [1.0],
            }
        }
        frames = candidate_frames(base, prior=[1 / 3, 1 / 3, 1 / 3], config=config, global_reference=global_reference)
        shrink = [frame for cid, params, frame in frames if cid == "p41_global_regime_shrinkage"][0]
        self.assertEqual(shrink.loc[0, "prob_down"], 1.0)
        self.assertEqual(shrink.loc[1, "prob_up"], 1.0)

    def test_phase41b_candidate_scope_is_explicit(self):
        base = pd.DataFrame(
            {
                "row_id": [1],
                "prob_down": [0.2],
                "prob_neutral": [0.5],
                "prob_up": [0.3],
                "score": [0.1],
                "pred_label": [0],
                "signal": [0],
            }
        )
        config = {
            "candidate_grids": {
                "probability_temperature": [1.0],
                "prior_blend_weight": [0.0],
                "global_regime_shrinkage": [0.0],
                "score_threshold": [0.05],
            }
        }
        frames = candidate_frames(base, prior=[1 / 3, 1 / 3, 1 / 3], config=config, global_reference=base)
        observed = {candidate_id for candidate_id, _, _ in frames}

        self.assertTrue(observed.issubset(PHASE41B_EXECUTED_CANDIDATES))
        self.assertTrue(PHASE41B_DEFERRED_CANDIDATES.isdisjoint(observed))
        self.assertIn("p41_score_threshold", PHASE41B_DEFERRED_CANDIDATES)


if __name__ == "__main__":
    unittest.main()
