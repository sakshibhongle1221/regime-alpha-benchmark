from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.fold_checkpoint import (
    FRAME_FILES,
    checkpoint_exists,
    initialize_run_state,
    load_fold_checkpoint,
    write_fold_checkpoint,
)


def example_frames() -> dict[str, pd.DataFrame]:
    return {
        "predictions": pd.DataFrame([{"fold": 1, "method": "global_lgbm", "score": 0.1}]),
        "assignments": pd.DataFrame([{"fold": 1, "method": "hmm", "regime": 2}]),
        "implementations": pd.DataFrame([{"fold": 1, "method": "hmm", "implementation": "causal"}]),
        "manifest": pd.DataFrame([{"fold": 1, "encoder_method": "guided", "model_hash": "a" * 64}]),
        "losses": pd.DataFrame([{"fold": 1, "encoder_method": "guided", "loss": 1.0}]),
        "coverage": pd.DataFrame([{"fold": 1, "method": "global_lgbm", "test_prediction_rows": 1}]),
    }


class FoldCheckpointTests(unittest.TestCase):
    def test_round_trip_validates_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            bounds = {"fold": 1, "outer_train_end": 100, "outer_test_start": 120}
            write_fold_checkpoint(run_dir, 1, "protocol", bounds, example_frames())
            self.assertTrue(checkpoint_exists(run_dir, 1))
            loaded = load_fold_checkpoint(run_dir, 1, "protocol", bounds)
            self.assertEqual(set(loaded), set(FRAME_FILES))
            self.assertEqual(len(loaded["predictions"]), 1)

    def test_tampered_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            bounds = {"fold": 1}
            directory = write_fold_checkpoint(
                run_dir, 1, "protocol", bounds, example_frames()
            )
            with (directory / "predictions.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "hash predictions"):
                load_fold_checkpoint(run_dir, 1, "protocol", bounds)

    def test_resume_rejects_changed_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            state = {
                "protocol_hash": "one",
                "config_hash": "config-one",
                "data_hash": "data",
                "source_hash": "source",
                "fold_hash": "folds",
            }
            initialize_run_state(run_dir, state, resume=False)
            changed = {**state, "protocol_hash": "two", "config_hash": "config-two"}
            with self.assertRaisesRegex(RuntimeError, "config_hash"):
                initialize_run_state(run_dir, changed, resume=True)

    def test_existing_run_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            state = {"protocol_hash": "one"}
            initialize_run_state(run_dir, state, resume=False)
            with self.assertRaisesRegex(RuntimeError, "Pass --resume"):
                initialize_run_state(run_dir, state, resume=False)

    def test_incomplete_write_is_replaced_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            incomplete = run_dir / "checkpoints" / "fold_01.writing"
            incomplete.mkdir(parents=True)
            (incomplete / "partial.txt").write_text("partial", encoding="utf-8")
            bounds = {"fold": 1}
            directory = write_fold_checkpoint(
                run_dir, 1, "protocol", bounds, example_frames()
            )
            self.assertFalse(incomplete.exists())
            metadata = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["fold"], 1)


if __name__ == "__main__":
    unittest.main()
