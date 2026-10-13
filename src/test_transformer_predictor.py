"""Saved Transformer smoke tests using held-out Not-labelled records."""

import unittest
from pathlib import Path

import joblib
import numpy as np

from src.predict_transformer import predict_antenatal_depression


class TransformerNotDepressedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        split = joblib.load(project_root / "models" / "data_splits.joblib")
        cls.features = split["X_test"]
        cls.labels = np.asarray(split["y_test"]).reshape(-1)

    def test_held_out_not_cases_return_valid_not_screening_results(self):
        not_indices = np.flatnonzero(self.labels == 0)[:10]
        self.assertEqual(len(not_indices), 10)

        results = [
            predict_antenatal_depression(
                self.features.iloc[int(index)].to_dict()
            )
            for index in not_indices
        ]

        self.assertTrue(
            all(result["label"] == "Not Depressed" for result in results)
        )
        self.assertTrue(
            all(0.0 <= result["probability"] <= 1.0 for result in results)
        )


if __name__ == "__main__":
    unittest.main()
