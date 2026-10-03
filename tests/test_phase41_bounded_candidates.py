import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase41_bounded_candidates import (
    blend_with_prior,
    negative_log_likelihood,
    normalize_probabilities,
    posterior_temperature_weights,
    select_by_inner_validation,
    temperature_scale_probabilities,
)


class Phase41BoundedCandidateTests(unittest.TestCase):
    def test_temperature_scaling_preserves_probability_rows(self):
        probs = np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2]])
        scaled = temperature_scale_probabilities(probs, temperature=1.5)
        self.assertEqual(scaled.shape, probs.shape)
        np.testing.assert_allclose(scaled.sum(axis=1), np.ones(2))
        self.assertTrue(np.all(scaled > 0))

    def test_prior_blend_moves_probabilities_toward_prior(self):
        probs = np.array([[0.9, 0.05, 0.05]])
        prior = np.array([0.3, 0.4, 0.3])
        blended = blend_with_prior(probs, prior, weight=0.5)
        np.testing.assert_allclose(blended.sum(axis=1), np.ones(1))
        self.assertLess(blended[0, 0], probs[0, 0])
        self.assertGreater(blended[0, 1], probs[0, 1])

    def test_posterior_temperature_weights_are_normalized(self):
        posteriors = np.array([[0.25, 0.25, 0.25, 0.25], [0.8, 0.1, 0.05, 0.05]])
        weights = posterior_temperature_weights(posteriors, temperature=0.75)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(2))
        self.assertGreater(weights[1, 0], posteriors[1, 0])

    def test_negative_log_likelihood_uses_label_mapping(self):
        probs = normalize_probabilities(np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]]))
        labels = np.array([-1, 1])
        nll = negative_log_likelihood(probs, labels, {-1: 0, 0: 1, 1: 2})
        self.assertLess(nll, 0.5)

    def test_select_by_inner_validation_enforces_guardrails(self):
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "bad_turnover",
                    "inner_validation_nll": 0.1,
                    "coverage_ok": True,
                    "turnover_increase_vs_baseline": 0.5,
                },
                {
                    "candidate_id": "safe_best",
                    "inner_validation_nll": 0.2,
                    "coverage_ok": True,
                    "turnover_increase_vs_baseline": 0.1,
                },
                {
                    "candidate_id": "safe_worse",
                    "inner_validation_nll": 0.3,
                    "coverage_ok": True,
                    "turnover_increase_vs_baseline": 0.0,
                },
            ]
        )
        selected = select_by_inner_validation(candidates)
        self.assertEqual(selected["candidate_id"], "safe_best")


if __name__ == "__main__":
    unittest.main()
