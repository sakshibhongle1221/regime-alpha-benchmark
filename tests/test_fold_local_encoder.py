import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import FEATURE_COLS, WINDOW_SIZE
from encoder import TemporalEncoder
from fold_local_encoder import (
    FoldEncoderConfig,
    build_training_windows,
    encode_causal_rows,
    fit_fold_encoder,
    fit_training_hmm,
    fit_training_scaler,
    make_fold_bounds,
    normalize_with_scaler,
    validate_fold_encoder_boundaries,
)


class FoldLocalEncoderBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        cls.matrices = {
            "AAAUSDT": rng.normal(size=(520, len(FEATURE_COLS))).astype(np.float32),
            "BBBUSDT": rng.normal(size=(520, len(FEATURE_COLS))).astype(np.float32),
        }
        cls.config = FoldEncoderConfig(
            epochs=1,
            batch_size=32,
            max_windows=64,
            inner_validation_bars=100,
            inner_embargo_bars=20,
            label_purge_bars=8,
            seed=7,
            device="cpu",
        )
        cls.bounds = make_fold_bounds(1, 400, 440, 500, cls.config)

    def test_future_rows_do_not_change_training_scaler(self):
        first = fit_training_scaler(self.matrices, 300)
        changed = {symbol: matrix.copy() for symbol, matrix in self.matrices.items()}
        for matrix in changed.values():
            matrix[300:] += 10_000.0
        second = fit_training_scaler(changed, 300)
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.std, second.std)

    def test_future_rows_do_not_change_training_hmm(self):
        scaler = fit_training_scaler(self.matrices, 300)
        normalized = normalize_with_scaler(self.matrices, scaler)
        first_model, first_labels, first_hash = fit_training_hmm(normalized, 300, 99)

        changed = {symbol: matrix.copy() for symbol, matrix in normalized.items()}
        for matrix in changed.values():
            matrix[300:] *= -50.0
        second_model, second_labels, second_hash = fit_training_hmm(changed, 300, 99)

        self.assertEqual(first_hash, second_hash)
        pd.testing.assert_frame_equal(first_labels, second_labels)
        np.testing.assert_allclose(
            first_model.transmat_, second_model.transmat_, rtol=1e-6, atol=1e-7
        )

    def test_training_and_validation_windows_respect_boundaries(self):
        normalized = normalize_with_scaler(
            self.matrices, fit_training_scaler(self.matrices, self.bounds.inner_train_end)
        )
        train_labels = pd.DataFrame(
            [
                {"symbol": symbol, "feat_idx": idx, "hmm_regime": idx % 4}
                for symbol in normalized
                for idx in range(self.bounds.inner_train_end)
            ]
        )
        validation_labels = pd.DataFrame(
            [
                {"symbol": symbol, "feat_idx": idx, "hmm_regime": idx % 4}
                for symbol in normalized
                for idx in range(
                    self.bounds.inner_validation_start,
                    self.bounds.inner_validation_end,
                )
            ]
        )
        train = build_training_windows(
            normalized,
            0,
            self.bounds.inner_train_end,
            0,
            1,
            hmm_labels=train_labels,
        )
        validation = build_training_windows(
            normalized,
            self.bounds.inner_validation_start,
            self.bounds.inner_validation_end,
            0,
            2,
            hmm_labels=validation_labels,
        )
        validate_fold_encoder_boundaries(train, validation, self.bounds, False)
        self.assertLess(int(train["feat_idx"].max()), self.bounds.inner_train_end)
        self.assertGreaterEqual(
            int(validation["start_idx"].min()), self.bounds.inner_validation_start
        )

    def test_vanilla_positive_window_stays_inside_boundary(self):
        normalized = normalize_with_scaler(
            self.matrices, fit_training_scaler(self.matrices, self.bounds.inner_train_end)
        )
        samples = build_training_windows(
            normalized,
            0,
            self.bounds.inner_train_end,
            0,
            1,
            vanilla_positive=True,
        )
        self.assertLessEqual(
            int(samples["feat_idx"].max()) + 1,
            self.bounds.inner_train_end - 1,
        )

    def test_causal_embedding_ignores_later_feature_mutation(self):
        scaler = fit_training_scaler(self.matrices, 300)
        model = TemporalEncoder(n_features=len(FEATURE_COLS), latent_dim=16)
        frame = pd.DataFrame(
            {
                "row_id": [0, 1, 2, 3],
                "symbol": ["AAAUSDT", "AAAUSDT", "BBBUSDT", "BBBUSDT"],
                "feat_idx": [100, 150, 100, 150],
            }
        )
        first = encode_causal_rows(model, self.matrices, scaler, frame, "cpu")
        changed = {symbol: matrix.copy() for symbol, matrix in self.matrices.items()}
        for matrix in changed.values():
            matrix[151:] += 999.0
        second = encode_causal_rows(model, changed, scaler, frame, "cpu")
        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)

    def test_repeat_training_reproduces_manifest_hashes(self):
        first = fit_fold_encoder("vanilla", self.matrices, self.bounds, self.config)
        second = fit_fold_encoder("vanilla", self.matrices, self.bounds, self.config)
        self.assertEqual(first.input_hash, second.input_hash)
        self.assertEqual(first.model_hash, second.model_hash)
        self.assertEqual(first.selected_epoch, second.selected_epoch)

    def test_window_before_history_is_rejected(self):
        scaler = fit_training_scaler(self.matrices, 300)
        model = TemporalEncoder(n_features=len(FEATURE_COLS), latent_dim=16)
        frame = pd.DataFrame(
            {"row_id": [0], "symbol": ["AAAUSDT"], "feat_idx": [WINDOW_SIZE - 2]}
        )
        with self.assertRaises(ValueError):
            encode_causal_rows(model, self.matrices, scaler, frame, "cpu")


if __name__ == "__main__":
    unittest.main()
