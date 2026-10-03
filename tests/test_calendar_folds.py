import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_models import (
    align_to_common_calendar,
    fold_ranges,
    row_ids_for_fold,
    validate_calendar_aligned_panel,
)


def make_panel(starts: dict[str, str], periods: int = 6_000) -> pd.DataFrame:
    rows = []
    row_id = 0
    for symbol, start in starts.items():
        for feat_idx, timestamp in enumerate(pd.date_range(start, periods=periods, freq="h")):
            rows.append(
                {
                    "row_id": row_id,
                    "symbol": symbol,
                    "feat_idx": feat_idx,
                    "open_time": timestamp,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


class CalendarFoldTests(unittest.TestCase):
    def test_common_calendar_reindexes_staggered_histories(self) -> None:
        frame = make_panel({"AAAUSDT": "2024-01-01", "BBBUSDT": "2024-02-01"})
        aligned = align_to_common_calendar(frame, ["AAAUSDT", "BBBUSDT"], warmup_rows=59)
        validate_calendar_aligned_panel(aligned)
        counts = aligned.groupby("symbol").size()
        self.assertEqual(counts.nunique(), 1)
        starts = aligned.groupby("symbol")["open_time"].min()
        self.assertEqual(starts.nunique(), 1)
        self.assertTrue((aligned.groupby("symbol")["feat_idx"].min() == 59).all())

    def test_staggered_histories_are_rejected(self) -> None:
        frame = make_panel({"AAAUSDT": "2024-01-01", "BBBUSDT": "2024-02-01"})
        with self.assertRaisesRegex(RuntimeError, "Unsafe multi-asset positional folds"):
            fold_ranges(frame)

    def test_aligned_histories_generate_calendar_safe_folds(self) -> None:
        frame = make_panel({"AAAUSDT": "2024-01-01", "BBBUSDT": "2024-01-01"})
        validate_calendar_aligned_panel(frame)
        folds = fold_ranges(frame)
        self.assertGreater(len(folds), 0)
        indexed = frame.set_index("row_id", drop=False)
        train_end, test_start, test_end = folds[0]
        train_ids, test_ids = row_ids_for_fold(indexed, train_end, test_start, test_end)
        self.assertLess(
            indexed.loc[train_ids, "open_time"].max(),
            indexed.loc[test_ids, "open_time"].min(),
        )

    def test_row_selector_rejects_calendar_overlap(self) -> None:
        frame = make_panel({"AAAUSDT": "2024-01-01", "BBBUSDT": "2024-02-01"})
        indexed = frame.set_index("row_id", drop=False)
        with self.assertRaisesRegex(RuntimeError, "Calendar leakage detected"):
            row_ids_for_fold(indexed, 4_320, 4_440, 5_160)


if __name__ == "__main__":
    unittest.main()
