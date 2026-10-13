"""Tests for raw-questionnaire Random Forest prediction."""

import unittest

from src.predict_random_forest import (
    get_model_info,
    predict_antenatal_depression,
)


SAMPLE_ANSWERS = {
    "Age": 26,
    "Gestational Age": 24,
    "Number of sons": 1,
    "Number of daughters": 1,
    "Total Number of Children": 2,
    "Gravida": "Multigravida",
    "Female Education": "Graduation",
    "Husband Education": "Graduation",
    "Working Status": "Housewife",
    "Physical Health": "Healthy",
    "Previous Miscarriage": "No",
    "Sufficient Money for Basic Needs": "Yes",
    "Current Appereance Acceptance": "Yes",
    "Family System": "Nuclear",
    "Male Gender Preference": "No",
    "Relationship with Mother in-law": "Good",
    "Little interest or pleasure in doing things": 0,
    "Feeling down, depressed, or hopeless": 0,
    "Trouble falling or staying sleep or sleeping too much": 0,
    "Feeling tired or having little energy": 0,
    "Poor appetite or overeating": 0,
    "Feeling badabout yourself that you are failure or have let yourself or your family down": 0,
    "Trouble concentrating on things, such as reading the newspaper or watching television": 0,
    "Moving or speaking so slowly that other people could have Noticed.": 0,
    "Thoughts that you would be better off dead, or of hurting yourself": 0,
}


class RandomForestPredictorTests(unittest.TestCase):
    def test_raw_answers_are_preprocessed_and_predicted(self):
        result = predict_antenatal_depression(SAMPLE_ANSWERS)

        self.assertEqual(len(SAMPLE_ANSWERS), 25)
        self.assertEqual(len(get_model_info()["feature_names"]), 59)
        self.assertIn(result["prediction"], (0, 1))
        self.assertIn(result["label"], ("Not Depressed", "Depressed"))
        self.assertEqual(
            set(result["class_probabilities"]),
            {"Not Depressed", "Depressed"},
        )
        self.assertAlmostEqual(sum(result["class_probabilities"].values()), 1.0)

    def test_missing_questionnaire_answer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            predict_antenatal_depression({})

    def test_unknown_category_is_rejected(self):
        answers = dict(SAMPLE_ANSWERS)
        answers["Working Status"] = "Unknown"

        with self.assertRaisesRegex(ValueError, "Unsupported category"):
            predict_antenatal_depression(answers)

    def test_schema_missing_category_uses_saved_imputer(self):
        answers = dict(SAMPLE_ANSWERS)
        answers["Male Gender Preference"] = "__MISSING__"

        result = predict_antenatal_depression(answers)

        self.assertIn("prediction", result)
        self.assertEqual(
            result["imputed_features"],
            ["Male Gender Preference"],
        )

    def test_missing_sentinel_is_not_allowed_for_other_categories(self):
        answers = dict(SAMPLE_ANSWERS)
        answers["Working Status"] = "__MISSING__"

        with self.assertRaisesRegex(ValueError, "Unsupported category"):
            predict_antenatal_depression(answers)

    def test_non_numeric_answer_is_rejected(self):
        answers = dict(SAMPLE_ANSWERS)
        answers["Age"] = "twenty-six"

        with self.assertRaisesRegex(ValueError, "must be numbers"):
            predict_antenatal_depression(answers)

    def test_boolean_numeric_answer_is_rejected(self):
        answers = dict(SAMPLE_ANSWERS)
        answers["Age"] = True

        with self.assertRaisesRegex(ValueError, "Boolean values"):
            predict_antenatal_depression(answers)


if __name__ == "__main__":
    unittest.main()
