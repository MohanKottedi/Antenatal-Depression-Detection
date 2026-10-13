"""Tests for training-data normalization and duplicate handling."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model import RAW_FEATURE_NAMES
from train_model import TARGET_COLUMN, load_dataset, split_indices


def make_record(age: int, label: str) -> dict[str, object]:
    record: dict[str, object] = {
        RAW_FEATURE_NAMES[0]: age,
        RAW_FEATURE_NAMES[1]: 24,
        RAW_FEATURE_NAMES[2]: 0,
        RAW_FEATURE_NAMES[3]: 0,
        RAW_FEATURE_NAMES[4]: 0,
        TARGET_COLUMN: label,
    }
    record.update({
        name: "test-category"
        for name in RAW_FEATURE_NAMES[5:]
    })
    return record


class TrainingDataTests(unittest.TestCase):
    def load_records(self, records: list[dict[str, object]]):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            return load_dataset(path)

    def test_duplicate_feature_and_label_rows_are_removed(self):
        first = make_record(25, "Not")
        second = make_record(30, "Depressed")

        features, target = self.load_records([first, first, second, second])

        self.assertEqual(len(features), 2)
        self.assertEqual(target.tolist(), [0, 1])

    def test_identical_features_with_conflicting_labels_are_preserved(self):
        features, target = self.load_records([
            make_record(25, "Not"),
            make_record(25, "Depressed"),
        ])

        self.assertEqual(len(features), 2)
        self.assertEqual(sorted(target.tolist()), [0, 1])

    def test_identical_model_input_profiles_stay_in_one_split(self):
        profile_ids = np.repeat(np.arange(80), 2)
        features = pd.DataFrame({"profile": profile_ids})
        target = np.repeat(
            np.repeat(np.array([0, 1]), 40),
            2,
        )

        train, validation, test = split_indices(features, target)
        groups = pd.util.hash_pandas_object(
            features,
            index=False,
        ).to_numpy()
        train_groups = set(groups[train])
        validation_groups = set(groups[validation])
        test_groups = set(groups[test])

        self.assertEqual(len(set(train) | set(validation) | set(test)), len(target))
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)


if __name__ == "__main__":
    unittest.main()
