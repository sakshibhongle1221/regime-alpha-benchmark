import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase43b_locked_holdout_registration import (
    DEFAULT_CONFIG,
    build_manifest,
    load_config,
    select_registered_symbols,
    validate_parent_freeze,
)


class Phase43BLockedHoldoutRegistrationTests(unittest.TestCase):
    def test_config_forbids_model_outcome_inputs(self):
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["data_role"], "locked_unobserved_registration_only")
        forbidden = set(config["forbidden_inputs"])
        self.assertIn("model_predictions", forbidden)
        self.assertIn("alpha_metrics", forbidden)
        self.assertIn("threshold_search_on_holdout", forbidden)
        self.assertIn("rerun_after_failure", forbidden)

    def test_parent_freeze_keeps_single_guided_hmm_candidate(self):
        config = load_config(DEFAULT_CONFIG)
        parent = validate_parent_freeze(config)
        self.assertEqual(parent["final_candidate"]["method"], "regime_lgbm_hmm_guided_hmm")

    def test_selection_uses_first_external_eligible_assets_only(self):
        config = load_config(DEFAULT_CONFIG)
        config["minimum_assets"] = 2
        quality = pd.DataFrame(
            [
                {"design_rank": 1, "symbol": "BTCUSDT", "holdout_eligible": False},
                {"design_rank": 21, "symbol": "FILUSDT", "holdout_eligible": True},
                {"design_rank": 22, "symbol": "ARBUSDT", "holdout_eligible": True},
                {"design_rank": 23, "symbol": "OPUSDT", "holdout_eligible": True},
            ]
        )
        selected = select_registered_symbols(config, quality)
        self.assertEqual(selected["symbol"].tolist(), ["FILUSDT", "ARBUSDT"])

    def test_manifest_blocks_when_minimum_assets_are_missing(self):
        config = load_config(DEFAULT_CONFIG)
        parent = {"final_candidate": {"method": "regime_lgbm_hmm_guided_hmm"}}
        quality = pd.DataFrame(
            [
                {
                    "design_rank": 21,
                    "symbol": "FILUSDT",
                    "external_candidate": True,
                    "ohlcv_rows": 0,
                    "feature_rows": 0,
                    "target_rows": 0,
                    "max_gap_hours": 0.0,
                    "holdout_eligible": False,
                    "failure_reason": "insufficient_ohlcv",
                }
            ]
        )
        selected = select_registered_symbols(config, quality)
        manifest = build_manifest(config, parent, quality, selected).set_index("item")
        self.assertEqual(manifest.loc["registration_status", "value"], "blocked_not_ready")
        self.assertEqual(manifest.loc["selected_asset_count", "value"], "0")


if __name__ == "__main__":
    unittest.main()
