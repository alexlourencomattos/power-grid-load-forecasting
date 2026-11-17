import unittest

import numpy as np
import pandas as pd

from ons_load_pipeline.features import FEATURE_COLUMNS, make_prediction_features


class FeatureTests(unittest.TestCase):
    def test_prediction_features_use_only_history_available_at_origin(self) -> None:
        index = pd.date_range("2025-01-01", periods=1000, freq="h")
        series = pd.Series(np.arange(1000, dtype=float), index=index)
        origin = index[-1]
        examples = make_prediction_features(series, origin)

        self.assertEqual(list(examples.features.columns), FEATURE_COLUMNS)
        self.assertEqual(len(examples.features), 168)
        self.assertFalse(examples.features.isna().any().any())
        self.assertTrue((examples.timestamps > origin).all())
        expected_first_weekly_lag = series.loc[origin + pd.Timedelta(hours=1 - 168)]
        self.assertEqual(examples.features.loc[0, "lag_168"], expected_first_weekly_lag)


if __name__ == "__main__":
    unittest.main()
