import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import information_coefficients, non_overlapping_returns, portfolio_metrics


def prediction_frame(periods: int = 48, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC")):
    rows = []
    times = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    for symbol_idx, symbol in enumerate(symbols):
        for idx, timestamp in enumerate(times):
            target = (idx - periods / 2) / periods + symbol_idx * 0.01
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "score": target,
                    "signal": 1 if target > 0 else -1,
                    "target_return": target / 100,
                }
            )
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def test_non_overlapping_grid_keeps_one_row_per_horizon(self):
        pred = prediction_frame()
        result = non_overlapping_returns(pred, 8, 0.001, rebalance_offset_utc=0)
        self.assertEqual(len(result), len(pred) // 8)
        for _, group in result.groupby("symbol"):
            self.assertTrue((group.open_time.diff().dropna() == pd.Timedelta(hours=8)).all())

    def test_execution_signal_is_lagged_before_grid_selection(self):
        pred = prediction_frame(periods=16, symbols=("AAA",))
        pred["signal"] = np.arange(len(pred))
        result = non_overlapping_returns(pred, 8, 0.0, rebalance_offset_utc=0)
        original = pred.set_index("open_time")["signal"]
        for row in result.itertuples():
            previous_time = row.open_time - pd.Timedelta(hours=1)
            expected = original.loc[previous_time] if previous_time in original.index else 0
            self.assertEqual(row.execution_signal, expected)

    def test_primary_ic_is_mean_asset_time_series_ic(self):
        pred = prediction_frame()
        metrics = information_coefficients(pred)
        self.assertAlmostEqual(metrics["IC"], metrics["mean_asset_IC"])
        self.assertAlmostEqual(metrics["mean_asset_IC"], 1.0)
        self.assertEqual(metrics["n_asset_IC"], 3)
        self.assertGreater(metrics["n_cross_sectional_IC"], 0)

    def test_sharpe_uses_non_overlapping_observations(self):
        pred = prediction_frame()
        metrics = portfolio_metrics(pred, 8, 0.001)
        self.assertEqual(metrics["n_return_observations"], 48 // 8)
        self.assertEqual(metrics["rebalance_offset_utc"], 0)

    def test_execution_positions_reset_at_fold_boundaries(self):
        first = prediction_frame(periods=16, symbols=("AAA",))
        second = prediction_frame(periods=16, symbols=("AAA",))
        second["open_time"] = second["open_time"] + pd.Timedelta(days=10)
        first["fold"] = 1
        second["fold"] = 2
        pred = pd.concat([first, second], ignore_index=True)
        result = non_overlapping_returns(pred, 8, 0.001)
        first_by_fold = result.sort_values("open_time").groupby("fold").first()
        self.assertTrue((first_by_fold["execution_signal"] == 0).all())

    def test_mixed_timezone_strings_are_normalized_to_utc(self):
        pred = prediction_frame(periods=16, symbols=("AAA",))
        pred.loc[:7, "open_time"] = pred.loc[:7, "open_time"].dt.tz_convert(
            "Asia/Kolkata"
        ).astype(str)
        pred.loc[8:, "open_time"] = pred.loc[8:, "open_time"].dt.tz_convert(
            "UTC"
        ).astype(str)
        result = non_overlapping_returns(pred, 8, 0.001)
        self.assertEqual(str(result["open_time"].dt.tz), "UTC")


if __name__ == "__main__":
    unittest.main()
