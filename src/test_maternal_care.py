import unittest
from datetime import date

from src.maternal_care import pregnancy_snapshot, urgent_symptoms


class PregnancyDatingTests(unittest.TestCase):
    def test_lmp_uses_standard_280_day_estimate_and_completed_weeks(self):
        snapshot = pregnancy_snapshot(
            date(2026, 1, 1),
            None,
            today=date(2026, 4, 2),
        )

        self.assertEqual(snapshot["estimated_due_date"], "2026-10-08")
        self.assertEqual(snapshot["due_date"], "2026-10-08")
        self.assertEqual(snapshot["gestational_week"], 13)
        self.assertEqual(snapshot["gestational_day"], 0)
        self.assertEqual(snapshot["trimester"], 1)

    def test_confirmed_due_date_takes_precedence_without_changing_lmp_estimate(self):
        snapshot = pregnancy_snapshot(
            date(2026, 1, 1),
            date(2026, 10, 20),
            today=date(2026, 4, 2),
        )

        self.assertEqual(snapshot["estimated_due_date"], "2026-10-08")
        self.assertEqual(snapshot["due_date"], "2026-10-20")
        self.assertEqual(snapshot["dating_source"], "last_menstrual_period")

    def test_confirmed_due_date_can_supply_estimated_week_when_lmp_is_unknown(self):
        snapshot = pregnancy_snapshot(
            None,
            date(2026, 10, 8),
            today=date(2026, 4, 2),
        )

        self.assertEqual(snapshot["estimated_due_date"], None)
        self.assertEqual(snapshot["gestational_week"], 13)
        self.assertEqual(snapshot["dating_source"], "clinically_confirmed_due_date")

    def test_missing_or_future_dates_do_not_invent_current_gestational_age(self):
        missing = pregnancy_snapshot(None, None, today=date(2026, 4, 2))
        future = pregnancy_snapshot(
            date(2026, 5, 1),
            None,
            today=date(2026, 4, 2),
        )

        self.assertIsNone(missing["gestational_week"])
        self.assertIsNone(future["gestational_week"])
        self.assertIsNone(future["progress_percentage"])


class UrgentSymptomTests(unittest.TestCase):
    def test_detects_urgent_maternal_warning_signs(self):
        for message in (
            "I have heavy bleeding",
            "I am having chest pain",
            "My baby's movements stopped",
            "I have a severe headache",
        ):
            with self.subTest(message=message):
                self.assertTrue(urgent_symptoms(message))

    def test_does_not_trigger_for_general_pregnancy_questions(self):
        self.assertFalse(urgent_symptoms("What foods are common in pregnancy?"))


if __name__ == "__main__":
    unittest.main()
