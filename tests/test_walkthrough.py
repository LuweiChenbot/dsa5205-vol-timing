"""Recording snapshot checks must not hide coverage or schema differences."""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from src.walkthrough import _same


class SnapshotComparisonTests(unittest.TestCase):
    def setUp(self):
        self.saved = pd.DataFrame(
            {"prediction": [.1, .2], "status": ["fit", "fit"]},
            index=pd.date_range("2025-01-02", periods=2, name="decision_date"),
        )

    def compare(self, new, saved=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return _same("forecasts", new, self.saved if saved is None else saved, **kwargs)

    def test_reordered_labels_are_aligned_only_after_coverage_checks(self):
        new = self.saved.iloc[::-1, ::-1]
        self.assertEqual(self.compare(new), 0.)

    def test_extra_missing_and_duplicate_rows_or_columns_are_rejected(self):
        extra = self.saved.iloc[:1].copy()
        extra.index = pd.DatetimeIndex(["2025-01-06"], name="decision_date")
        extra["prediction"] = 999.
        cases = {
            "extra row": pd.concat([self.saved, extra]),
            "missing row": self.saved.iloc[:-1],
            "duplicate row": pd.concat([self.saved, self.saved.iloc[:1]]),
            "extra column": self.saved.assign(unexpected_signal=999.),
            "missing column": self.saved.drop(columns="status"),
            "duplicate column": pd.concat([self.saved, self.saved[["prediction"]]], axis=1),
            "same size wrong row": self.saved.rename(index={self.saved.index[0]: extra.index[0]}),
            "same size wrong column": self.saved.rename(columns={"prediction": "another_signal"}),
        }
        for label, new in cases.items():
            with self.subTest(label=label), self.assertRaises(AssertionError):
                self.compare(new)

    def test_invalid_reference_labels_are_also_rejected(self):
        duplicate = pd.concat([self.saved, self.saved.iloc[:1]])
        with self.assertRaises(AssertionError):
            self.compare(duplicate, duplicate)

    def test_numeric_values_use_absolute_tolerance_and_metadata_is_exact(self):
        new = self.saved.copy()
        new.loc[new.index[0], "prediction"] += 5e-11
        self.assertAlmostEqual(self.compare(new, tol=1e-10), 5e-11, places=15)
        with self.assertRaises(AssertionError):
            self.compare(new, tol=1e-12)
        with self.assertRaises(AssertionError):
            self.compare(self.saved.assign(status="fallback"))
        with self.assertRaises(AssertionError):
            self.compare(self.saved.assign(prediction=self.saved.prediction.astype(str)))

    def test_matching_nan_values_are_allowed_but_missingness_changes_fail(self):
        saved = self.saved.copy()
        saved["prediction"] = np.nan
        self.assertEqual(self.compare(saved.copy(), saved), 0.)
        with self.assertRaises(AssertionError):
            self.compare(saved.assign(prediction=0.), saved)
        new = self.saved.copy()
        new.loc[new.index[0], "prediction"] = np.nan
        with self.assertRaises(AssertionError):
            self.compare(new)

    def test_infinite_values_are_rejected_even_when_they_match(self):
        for value in [np.inf, -np.inf]:
            with self.subTest(value=value):
                invalid = self.saved.assign(prediction=value)
                with self.assertRaises(AssertionError):
                    self.compare(invalid, invalid)

    def test_empty_and_non_numeric_tables_compare_without_nanmax_errors(self):
        empty = self.saved.iloc[:0]
        self.assertEqual(self.compare(empty.copy(), empty), 0.)
        text = self.saved[["status"]]
        self.assertEqual(self.compare(text.copy(), text), 0.)

    def test_tolerance_must_be_nonnegative_and_finite(self):
        for tolerance in [-1., np.nan, np.inf]:
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.compare(self.saved.copy(), tol=tolerance)


if __name__ == "__main__":
    unittest.main()
