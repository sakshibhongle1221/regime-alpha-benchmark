import sys
import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freeze_development_dataset import (
    canonical_hash,
    comparable_summary_for_verification,
    load_config,
)
from fold_local_encoder_walkforward import verify_frozen_dataset


class DevelopmentFreezeTests(unittest.TestCase):
    def test_frozen_config_has_exact_unique_crypto20(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "crypto20_development_freeze_v1.json"
        config = load_config(path)
        self.assertEqual(len(config["symbols"]), 20)
        self.assertEqual(len(set(config["symbols"])), 20)
        self.assertEqual(config["data_role"], "development_observed")

    def test_canonical_hash_ignores_dictionary_key_order(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_freeze_boundaries_are_ordered(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "crypto20_development_freeze_v1.json"
        config = load_config(path)
        self.assertLess(config["feature_context_start"], config["prediction_start"])
        self.assertLess(config["prediction_start"], config["prediction_end"])

    def test_verification_ignores_only_database_file_hash_provenance(self):
        original = {
            "freeze_id": "crypto20-development-v1",
            "experiment_data_sha256": "stable-data",
            "symbol_manifest_sha256": "stable-symbols",
            "database_sha256": "old-shared-db-file",
        }
        expanded_database = {
            **original,
            "database_sha256": "new-shared-db-file",
        }
        changed_data = {
            **original,
            "experiment_data_sha256": "changed-data",
        }
        self.assertEqual(
            comparable_summary_for_verification(original),
            comparable_summary_for_verification(expanded_database),
        )
        self.assertNotEqual(
            comparable_summary_for_verification(original),
            comparable_summary_for_verification(changed_data),
        )

    def test_training_rejects_frozen_data_hash_drift(self):
        manifest = {
            "freeze_id": "test-freeze",
            "data_role": "development_observed",
            "symbols": ["AAA"],
            "experiment_data_sha256": "expected",
            "prediction_rows": 3,
            "fold_count": 1,
        }
        frame = pd.DataFrame({"row_id": [0, 1, 2]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            verify_frozen_dataset(path, ["AAA"], frame, [(1, 2, 3)], "expected")
            with self.assertRaisesRegex(RuntimeError, "data hash"):
                verify_frozen_dataset(path, ["AAA"], frame, [(1, 2, 3)], "changed")


if __name__ == "__main__":
    unittest.main()
