import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase43_locked_holdout_freeze import (
    DEFAULT_CONFIG,
    build_claim_rules,
    build_holdout_rules,
    load_config,
)


class Phase43LockedHoldoutFreezeTests(unittest.TestCase):
    def test_config_freezes_guided_hmm_and_excludes_rescue_paths(self):
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["data_role"], "locked_unobserved_until_phase43b")
        self.assertEqual(config["final_candidate"]["method"], "regime_lgbm_hmm_guided_hmm")
        excluded = set(config["excluded_from_final_candidate"])
        self.assertIn("phase41b_probability_calibration", excluded)
        self.assertIn("phase41b_soft_gating", excluded)
        self.assertIn("score_threshold_execution_control", excluded)

    def test_holdout_rules_forbid_selection_on_holdout(self):
        config = load_config(DEFAULT_CONFIG)
        rules = build_holdout_rules(config).set_index("rule_id")
        self.assertEqual(rules.loc["candidate_selection_on_holdout", "rule_value"], "forbidden")
        self.assertEqual(rules.loc["threshold_selection_on_holdout", "rule_value"], "forbidden")
        self.assertEqual(rules.loc["rerun_after_failure", "rule_value"], "forbidden")
        self.assertEqual(rules.loc["preferred_holdout", "rule_value"], "external_asset_holdout")

    def test_claim_rules_include_failure_reporting(self):
        config = load_config(DEFAULT_CONFIG)
        claims = build_claim_rules(config)
        text = " ".join(claims["rule"].astype(str))
        self.assertIn("failed locked confirmation", text)
        self.assertIn("tradable strategy", text)
        self.assertIn("threshold tuning rescued the model", text)


if __name__ == "__main__":
    unittest.main()
