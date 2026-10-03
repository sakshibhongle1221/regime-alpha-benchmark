import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fold_local_encoder_walkforward import verify_frozen_dataset
from phase43b_locked_holdout_freeze import FREEZE_ID, frame_hash


class Phase43BLockedHoldoutFreezeTests(unittest.TestCase):
    def test_frame_hash_is_stable_for_same_rows(self):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        self.assertEqual(frame_hash(frame, ["a", "b"]), frame_hash(frame.copy(), ["a", "b"]))

    def test_frozen_dataset_verifier_accepts_locked_registered_role(self):
        symbols = ["FILUSDT"]
        frame = pd.DataFrame({"x": [1, 2, 3]})
        folds = [(1, 2, 3)]
        data_hash = "abc"
        path = Path(".tmp") / "test_phase43b_locked_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "{"
            f'"freeze_id":"{FREEZE_ID}",'
            '"data_role":"locked_registered_unobserved",'
            '"symbols":["FILUSDT"],'
            '"experiment_data_sha256":"abc",'
            '"prediction_rows":3,'
            '"fold_count":1'
            "}",
            encoding="utf-8",
        )
        try:
            manifest = verify_frozen_dataset(path, symbols, frame, folds, data_hash)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(manifest["data_role"], "locked_registered_unobserved")


if __name__ == "__main__":
    unittest.main()
