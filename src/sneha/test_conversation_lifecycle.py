"""Deterministic tests for Sneha's conversation lifecycle."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

from src.sneha.conversation_engine import ConversationEngine
from src.sneha.conversation_manager import ConversationManager
from src.sneha.feature_extractor import validate_extraction
from src.sneha.feature_schema import FEATURE_SCHEMA, SYMPTOM_FEATURES
from src.sneha.safety import may_indicate_self_harm


ANSWERS = {
    "Age": ("26", 26),
    "Gestational Age": ("24 weeks", 24),
    "Number of sons": ("0", 0),
    "Number of daughters": ("0", 0),
    "Total Number of Children": ("0", 0),
    "Gravida": ("This is my first pregnancy", "Primigravida"),
    "Female Education": ("Graduation", "Graduation"),
    "Husband Education": ("Graduation", "Graduation"),
    "Working Status": ("I am a housewife", "Housewife"),
    "Physical Health": ("Healthy", "Healthy"),
    "Previous Miscarriage": ("No", "No"),
    "Sufficient Money for Basic Needs": ("Yes", "Yes"),
    "Current Appereance Acceptance": ("Yes", "Yes"),
    "Family System": ("Nuclear", "Nuclear"),
    "Male Gender Preference": ("No", "No"),
    "Relationship with Mother in-law": ("Good", "Good"),
    **{feature: ("Not at all", 0) for feature in SYMPTOM_FEATURES},
}


class StubLLM:
    def __init__(self):
        self.pending_feature = None
        self.chat = Mock(return_value="Could you tell me a little more?")
        self.extract_json = Mock(side_effect=self._extract)
        self.invalid_value = False

    def _extract(self, **_kwargs):
        feature = self.pending_feature()
        message, value = ANSWERS[feature]
        if self.invalid_value:
            value = "not a valid answer"

        return {
            "updates": [
                {
                    "feature": feature,
                    "value": value,
                    "confidence": 1.0,
                    "evidence": message,
                }
            ],
            "next_question": "",
        }


class ConversationLifecycleTests(unittest.TestCase):
    def make_engine(self):
        llm = StubLLM()
        engine = ConversationEngine(llm_client=llm)
        llm.pending_feature = engine.manager.get_pending_feature
        return engine, llm

    def test_normal_answer_updates_feature_and_progress(self):
        engine, _ = self.make_engine()
        started = engine.start()

        result = engine.respond("26")

        self.assertEqual(started["pending_feature"], "Age")
        self.assertEqual(result["updates"][0]["value"], 26)
        self.assertEqual(result["progress"]["collected"], 1)
        self.assertEqual(result["stage"], "background")
        self.assertEqual(result["pending_feature"], "Gestational Age")
        self.assertIsNotNone(started["progress"])
        self.assertIsNotNone(started["stage"])
        self.assertIsNotNone(result["progress"])
        self.assertIsNotNone(result["stage"])

    def test_question_templates_do_not_call_llm(self):
        engine, llm = self.make_engine()

        started = engine.start()

        self.assertEqual(started["assistant_message"], "May I ask how old you are?")
        llm.chat.assert_not_called()

    def test_gestational_age_days_are_converted_to_weeks_without_llm(self):
        engine, llm = self.make_engine()
        engine.start()
        engine.respond("26")
        llm.extract_json.reset_mock()

        result = engine.respond("15 days")

        self.assertAlmostEqual(
            engine.manager.features["Gestational Age"],
            15 / 7,
        )
        self.assertEqual(result["progress"]["collected"], 2)
        self.assertEqual(result["pending_feature"], "Number of sons")
        llm.extract_json.assert_not_called()
        llm.chat.assert_not_called()

    def test_explicit_weeks_and_days_are_converted_to_decimal_weeks(self):
        from src.sneha.feature_extractor import parse_gestational_age_weeks

        self.assertAlmostEqual(parse_gestational_age_weeks("15 days"), 15 / 7)
        self.assertAlmostEqual(
            parse_gestational_age_weeks("2 weeks and 1 day"),
            15 / 7,
        )
        self.assertEqual(parse_gestational_age_weeks("15"), None)

    def test_invalid_answer_does_not_advance_or_increase_progress(self):
        engine, llm = self.make_engine()
        engine.start()
        llm.invalid_value = True

        result = engine.respond("twenty-six")

        self.assertEqual(result["updates"], [])
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertEqual(result["stage"], "background")
        self.assertEqual(result["pending_feature"], "Age")
        self.assertFalse(result["completed"])
        self.assertIsNotNone(result["progress"])
        self.assertIsNotNone(result["stage"])
        self.assertIn("didn't quite catch", result["assistant_message"])

    def test_empty_extraction_keeps_pending_feature_and_progress(self):
        engine, _ = self.make_engine()
        engine.start()
        engine.llm.extract_json = Mock(return_value={"updates": []})

        result = engine.respond("I'm not sure")

        self.assertEqual(result["updates"], [])
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertEqual(result["pending_feature"], "Age")
        self.assertFalse(result["completed"])
        self.assertIsNotNone(result["stage"])

    def test_high_confidence_grounded_suggestion_requires_confirmation(self):
        engine, _ = self.make_engine()
        engine.start()
        engine.llm.extract_json = Mock(return_value={
            "updates": [],
            "suggested_update": {
                "feature": "Age",
                "value": 26,
                "confidence": 0.91,
                "evidence": "twenty-six",
            },
        })

        suggestion = engine.respond("I think I'm twenty-six, maybe.")

        self.assertTrue(suggestion["confirmation_pending"])
        self.assertEqual(suggestion["pending_feature"], "Age")
        self.assertEqual(suggestion["progress"]["collected"], 0)
        self.assertNotIn("Age", engine.manager.features)
        self.assertIn("26", suggestion["assistant_message"])

        confirmed = engine.respond("Yes")

        self.assertEqual(confirmed["updates"][0]["value"], 26)
        self.assertEqual(confirmed["progress"]["collected"], 1)
        self.assertEqual(confirmed["pending_feature"], "Gestational Age")
        self.assertNotIn("confirmation_pending", confirmed)
        engine.llm.extract_json.assert_called_once()

    def test_declining_suggestion_discards_it_and_repeats_feature(self):
        engine, llm = self.make_engine()
        engine.start()
        llm.extract_json = Mock(return_value={
            "updates": [],
            "suggested_update": {
                "feature": "Age",
                "value": 26,
                "confidence": 0.91,
                "evidence": "twenty-six",
            },
        })

        original_question = engine.manager.pending_question
        engine.respond("I think I'm twenty-six, maybe.")
        declined = engine.respond("No")

        self.assertFalse(declined["confirmation_pending"])
        self.assertEqual(declined["pending_feature"], "Age")
        self.assertEqual(declined["progress"]["collected"], 0)
        self.assertNotIn("Age", engine.manager.features)
        self.assertIn(original_question, declined["assistant_message"])
        llm.extract_json.assert_called_once()

    def test_unclear_confirmation_does_not_accept_or_discard_suggestion(self):
        engine, llm = self.make_engine()
        engine.start()
        llm.extract_json = Mock(return_value={
            "updates": [],
            "suggested_update": {
                "feature": "Age",
                "value": 26,
                "confidence": 0.91,
                "evidence": "twenty-six",
            },
        })

        engine.respond("I think I'm twenty-six, maybe.")
        unclear = engine.respond("I'm not sure")

        self.assertTrue(unclear["confirmation_pending"])
        self.assertEqual(unclear["pending_feature"], "Age")
        self.assertEqual(unclear["progress"]["collected"], 0)
        self.assertNotIn("Age", engine.manager.features)
        self.assertIn("yes or no", unclear["assistant_message"])
        llm.extract_json.assert_called_once()

    def test_low_confidence_suggestion_is_not_offered(self):
        engine, _ = self.make_engine()
        engine.start()
        engine.llm.extract_json = Mock(return_value={
            "updates": [],
            "suggested_update": {
                "feature": "Age",
                "value": 26,
                "confidence": 0.79,
                "evidence": "26",
            },
        })

        result = engine.respond("I'm maybe 26")

        self.assertNotIn("confirmation_pending", result)
        self.assertIsNone(engine.pending_confirmation)
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertEqual(result["pending_feature"], "Age")
        self.assertIn("didn't quite catch", result["assistant_message"])

    def test_unsupported_suggestion_is_not_offered(self):
        engine, _ = self.make_engine()
        engine.start()
        engine.llm.extract_json = Mock(return_value={
            "updates": [],
            "suggested_update": {
                "feature": "Age",
                "value": 26,
                "confidence": 0.99,
                "evidence": "not sure",
            },
        })

        result = engine.respond("I'm not sure")

        self.assertNotIn("confirmation_pending", result)
        self.assertIsNone(engine.pending_confirmation)
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertEqual(result["pending_feature"], "Age")
        self.assertIn("didn't quite catch", result["assistant_message"])

    def test_categorical_options_are_shown_after_three_unclear_answers(self):
        engine, llm = self.make_engine()
        engine.start()
        engine.manager.set_pending_feature(
            "Working Status",
            "Are you currently working?",
        )
        llm.extract_json = Mock(return_value={"updates": []})

        for _ in range(2):
            result = engine.respond("I'm not sure")
            self.assertNotIn("answer_options", result)
        result = engine.respond("I don't know")

        self.assertEqual(result["clarification_attempts"], 3)
        self.assertEqual(
            result["answer_options"],
            ["Housewife", "Working Lady"],
        )
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertEqual(result["pending_feature"], "Working Status")
        self.assertEqual(llm.extract_json.call_count, 3)

        selected = engine.respond("Housewife")

        self.assertEqual(selected["updates"][0]["value"], "Housewife")
        self.assertEqual(selected["progress"]["collected"], 1)
        self.assertNotIn("answer_options", selected)
        self.assertNotIn("Working Status", engine.clarification_attempts)
        self.assertEqual(llm.extract_json.call_count, 3)

    def test_numeric_question_shows_examples_after_three_unclear_answers(self):
        engine, llm = self.make_engine()
        engine.start()
        llm.extract_json = Mock(return_value={"updates": []})

        for _ in range(3):
            result = engine.respond("I don't know")

        self.assertEqual(result["clarification_attempts"], 3)
        self.assertNotIn("answer_options", result)
        self.assertIn("age in completed years", result["answer_hint"])
        self.assertIn("for example, 26", result["assistant_message"])
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertEqual(result["pending_feature"], "Age")

    def test_symptom_options_map_to_scale_and_not_diagnosis(self):
        engine, llm = self.make_engine()
        engine.start()
        feature = SYMPTOM_FEATURES[0]
        engine.manager.set_pending_feature(feature, "How often?")
        llm.extract_json = Mock(return_value={"updates": []})

        for _ in range(3):
            result = engine.respond("I don't know")

        self.assertEqual(
            result["answer_options"],
            ["Not at all", "Several days", "More than half the days", "Nearly every day"],
        )
        selected = engine.respond("Several days")

        self.assertEqual(selected["updates"][0]["value"], 1)
        self.assertEqual(selected["progress"]["collected"], 1)
        self.assertEqual(llm.extract_json.call_count, 3)

    def test_positive_self_harm_option_starts_safety_follow_up(self):
        engine, llm = self.make_engine()
        engine.start()
        feature = SYMPTOM_FEATURES[-1]
        engine.manager.set_pending_feature(feature, "How often?")
        llm.extract_json = Mock(return_value={"updates": []})

        for _ in range(3):
            result = engine.respond("I don't know")
        self.assertEqual(len(result["answer_options"]), 4)

        safety = engine.respond("Several days")

        self.assertTrue(safety["screening_stopped"])
        self.assertTrue(safety["safety_follow_up"])
        self.assertEqual(safety["progress"]["collected"], 0)
        self.assertNotIn(feature, engine.manager.features)
        self.assertEqual(llm.extract_json.call_count, 3)

    def test_empty_answer_keeps_progress_and_stage_defined(self):
        engine, _ = self.make_engine()
        engine.start()

        result = engine.respond("   ")

        self.assertEqual(result["progress"]["total"], len(FEATURE_SCHEMA))
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertIsNotNone(result["stage"])
        self.assertEqual(result["pending_feature"], "Age")

    def test_unrelated_repeat_does_not_count_twice_or_overwrite_answer(self):
        engine, _ = self.make_engine()
        engine.start()
        first = engine.respond("26")
        self.assertEqual(first["progress"]["collected"], 1)

        engine.llm.extract_json = Mock(return_value={
            "updates": [{
                "feature": "Age",
                "value": 35,
                "confidence": 1.0,
                "evidence": "26",
            }],
            "next_question": "",
        })
        repeated = engine.respond("26")

        self.assertEqual(repeated["updates"], [])
        self.assertEqual(repeated["progress"]["collected"], 1)
        self.assertEqual(engine.manager.features["Age"], 26)
        self.assertEqual(repeated["pending_feature"], "Gestational Age")

    def test_invalid_or_empty_extraction_preserves_accepted_values(self):
        engine, _ = self.make_engine()
        engine.start()
        engine.respond("26")
        engine.llm.extract_json = Mock(return_value={
            "updates": [{
                "feature": "Gestational Age",
                "value": 50,
                "confidence": 1.0,
                "evidence": "I am 50 weeks pregnant",
            }],
            "next_question": "",
        })

        result = engine.respond("I am 50 weeks pregnant")

        self.assertEqual(result["updates"], [])
        self.assertEqual(result["progress"]["collected"], 1)
        self.assertEqual(engine.manager.features["Age"], 26)
        self.assertNotIn("Gestational Age", engine.manager.features)
        self.assertEqual(result["pending_feature"], "Gestational Age")

    def test_numeric_value_not_supported_by_answer_is_rejected(self):
        engine, _ = self.make_engine()
        engine.start()
        engine.llm.extract_json = Mock(return_value={
            "updates": [{
                "feature": "Age",
                "value": 35,
                "confidence": 1.0,
                "evidence": "26",
            }],
            "next_question": "",
        })

        result = engine.respond("I am 26")

        self.assertEqual(result["updates"], [])
        self.assertNotIn("Age", engine.manager.features)
        self.assertEqual(result["pending_feature"], "Age")

    def test_symptom_score_must_match_patient_frequency(self):
        feature = "Feeling tired or having little energy"
        result = validate_extraction(
            {
                "updates": [{
                    "feature": feature,
                    "value": 3,
                    "confidence": 1.0,
                    "evidence": "Not at all",
                }],
            },
            patient_message="Not at all",
        )

        self.assertEqual(result["updates"], [])

    def test_symptom_suggestion_requires_explicit_frequency(self):
        feature = "Feeling tired or having little energy"
        result = validate_extraction(
            {
                "updates": [],
                "suggested_update": {
                    "feature": feature,
                    "value": 3,
                    "confidence": 0.99,
                    "evidence": "very tired",
                },
            },
            patient_message="I have been very tired.",
        )

        self.assertIsNone(result["suggested_update"])

    def test_symptom_confirmation_uses_frequency_label(self):
        feature = "Feeling tired or having little energy"
        self.assertEqual(
            ConversationEngine._confirmation_value({
                "feature": feature,
                "value": 2,
            }),
            "More than half the days",
        )

    def test_number_words_in_answer_are_supported(self):
        engine, _ = self.make_engine()
        engine.start()
        engine.llm.extract_json = Mock(return_value={
            "updates": [{
                "feature": "Age",
                "value": 26,
                "confidence": 1.0,
                "evidence": "twenty-six",
            }],
            "next_question": "",
        })

        result = engine.respond("I am twenty-six")

        self.assertEqual(result["updates"][0]["value"], 26)
        self.assertEqual(result["progress"]["collected"], 1)
        self.assertEqual(result["pending_feature"], "Gestational Age")

    def test_explicit_total_is_accepted_when_component_counts_differ(self):
        manager = ConversationManager()
        manager.features = {
            "Number of sons": 2,
            "Number of daughters": 2,
        }

        result = manager.update_features(
            {
                "updates": [{
                    "feature": "Total Number of Children",
                    "value": 3,
                    "confidence": 1.0,
                    "evidence": "total i have 3 childrens",
                }],
            },
            patient_message="total i have 3 childrens",
        )

        self.assertEqual(len(result["updates"]), 1)
        self.assertEqual(
            manager.features["Total Number of Children"],
            3,
        )
        self.assertEqual(manager.get_progress()["collected"], 3)

    def test_complete_interview_runs_transformer_once(self):
        engine, _ = self.make_engine()
        engine.start()
        expected = {
            "status": "success",
            "model_info": {"model": "Tabular Transformer Experiment A"},
            "prediction": {
                "prediction": 0,
                "label": "Not Depressed",
            },
        }

        with patch.object(
            engine,
            "predict_result",
            return_value=expected,
        ) as predict:
            while not engine.manager.completed:
                feature = engine.manager.get_pending_feature()
                message, _ = ANSWERS[feature]
                result = engine.respond(message)

        predict.assert_called_once_with()
        self.assertTrue(result["completed"])
        self.assertEqual(result["progress"]["collected"], 25)
        self.assertEqual(result["stage"], "completed")
        self.assertEqual(result["screening_result"], expected)
        self.assertIn("not a diagnosis", result["assistant_message"])
        self.assertIsNone(result["pending_feature"])
        subsequent = engine.respond("This is another answer.")
        self.assertTrue(subsequent["completed"])
        predict.assert_called_once_with()

    def test_incomplete_interview_does_not_attempt_prediction(self):
        engine, _ = self.make_engine()
        engine.start()

        result = engine.predict_result()

        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["prediction"])
        self.assertEqual(result["progress"]["collected"], 0)
        self.assertFalse(result["completed"])

    def test_prediction_attempt_is_cached_after_valid_completion(self):
        engine, _ = self.make_engine()
        engine.manager.features = {
            feature: ANSWERS[feature][1]
            for feature in FEATURE_SCHEMA
        }
        engine.manager.determine_stage()
        predictor = types.ModuleType("src.predict_transformer")
        predict = Mock(return_value={
            "prediction": 1,
            "label": "Depressed",
            "probability": 0.8,
        })
        predictor.predict_antenatal_depression = predict
        predictor.get_model_info = lambda: {"model": "Tabular Transformer"}

        with patch.dict(sys.modules, {"src.predict_transformer": predictor}):
            first = engine.predict_result()
            second = engine.predict_result()

        predict.assert_called_once()
        self.assertEqual(first["status"], "success")
        self.assertAlmostEqual(first["class_probabilities"]["Depressed"], 0.8)
        self.assertAlmostEqual(
            first["class_probabilities"]["Not Depressed"],
            0.2,
        )
        self.assertEqual(second, first)

    def test_prediction_validation_failure_is_reported(self):
        engine, _ = self.make_engine()
        engine.manager.features = {
            feature: ANSWERS[feature][1]
            for feature in FEATURE_SCHEMA
        }
        engine.manager.determine_stage()
        predictor = types.ModuleType("src.predict_transformer")

        def fail_prediction(_features):
            raise ValueError("invalid model inputs")

        predictor.predict_antenatal_depression = fail_prediction
        predictor.get_model_info = lambda: {"model": "Tabular Transformer"}

        with patch.dict(sys.modules, {"src.predict_transformer": predictor}):
            result = engine.predict_result()

        self.assertEqual(result["status"], "invalid_features")
        self.assertEqual(result["message"], "invalid model inputs")

    def test_prediction_failure_is_included_in_completion_response(self):
        engine, _ = self.make_engine()
        engine.start()
        failure = {
            "status": "invalid_features",
            "message": "saved preprocessor rejected the input",
        }

        with patch.object(
            engine,
            "predict_result",
            return_value=failure,
        ):
            while not engine.manager.completed:
                feature = engine.manager.get_pending_feature()
                message, _ = ANSWERS[feature]
                result = engine.respond(message)

        self.assertEqual(result["screening_result"], failure)
        self.assertIn(failure["message"], result["assistant_message"])
        self.assertNotIn("screening calculation. This result", result["assistant_message"])

    def test_safety_disclosure_stops_screening_without_prediction(self):
        engine, llm = self.make_engine()
        engine.start()

        with patch.object(engine, "predict_result") as predict:
            disclosure = engine.respond("I wish I were dead")
            follow_up = engine.respond("No, I am safe right now.")
            later = engine.respond("26")

        self.assertTrue(disclosure["safety_follow_up"])
        self.assertTrue(disclosure["screening_stopped"])
        self.assertIsNotNone(disclosure["progress"])
        self.assertIsNotNone(disclosure["stage"])
        self.assertEqual(disclosure["progress"]["collected"], 0)
        self.assertIsNotNone(follow_up["progress"])
        self.assertIsNotNone(follow_up["stage"])
        self.assertFalse(follow_up["safety_follow_up"])
        self.assertTrue(follow_up["screening_stopped"])
        self.assertTrue(later["screening_stopped"])
        self.assertFalse(engine.manager.completed)
        self.assertEqual(engine.manager.get_collected_features(), {})
        llm.extract_json.assert_not_called()
        predict.assert_not_called()

    def test_clear_denial_is_not_a_risk_trigger_but_mixed_disclosure_is(self):
        self.assertFalse(
            may_indicate_self_harm("I do not have suicidal thoughts.")
        )
        self.assertTrue(
            may_indicate_self_harm(
                "I have no suicidal thoughts, but I want to die."
            )
        )

    def test_manager_enters_safety_stage_for_final_required_feature(self):
        manager = ConversationManager()
        manager.features = {
            feature: ANSWERS[feature][1] for feature in FEATURE_SCHEMA
            if feature != (
                "Thoughts that you would be better off dead, "
                "or of hurting yourself"
            )
        }

        self.assertEqual(manager.determine_stage(), "safety")


if __name__ == "__main__":
    unittest.main()
