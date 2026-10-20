"""Direct endpoint tests for screening forms and voice synthesis."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import app as api
from fastapi import HTTPException
from src.sneha.feature_extractor import CATEGORY_VALUES
from src.sneha.feature_schema import (
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA,
    NUMERIC_FEATURES,
    SYMPTOM_FEATURES,
)


def valid_answers():
    answers = {}
    for feature, schema in FEATURE_SCHEMA.items():
        if feature in NUMERIC_FEATURES:
            answers[feature] = {
                "Age": 25,
                "Gestational Age": 20,
                "Number of sons": 0,
                "Number of daughters": 0,
                "Total Number of Children": 0,
            }[feature]
        elif feature in CATEGORICAL_FEATURES:
            answers[feature] = next(
                option
                for option in CATEGORY_VALUES[feature]
                if option != "__MISSING__"
            )
        else:
            answers[feature] = 0
    return answers


class FastAPIFormTests(unittest.TestCase):
    def test_schema_returns_all_validated_features_and_choices(self):
        schema = api.screening_form_schema()

        self.assertEqual(len(schema["fields"]), 25)
        self.assertEqual(
            {field["name"] for field in schema["fields"]},
            set(FEATURE_SCHEMA),
        )
        symptom_field = next(
            field
            for field in schema["fields"]
            if field["name"] == SYMPTOM_FEATURES[0]
        )
        self.assertEqual(
            [option["value"] for option in symptom_field["options"]],
            [0, 1, 2, 3],
        )

    def test_valid_answers_return_model_result(self):
        predictor = types.ModuleType("src.predict_transformer")
        predictor.predict_antenatal_depression = lambda _answers: {
            "label": "Depressed",
            "probability": 0.7,
        }
        predictor.get_model_info = lambda: {"model": "Test model"}

        with patch.dict(sys.modules, {"src.predict_transformer": predictor}):
            result = api.predict_from_form(
                api.ScreeningFormSubmission(answers=valid_answers())
            )

        self.assertEqual(result["screening_result"]["status"], "success")
        probabilities = result["screening_result"]["class_probabilities"]
        self.assertAlmostEqual(probabilities["Not Depressed"], 0.3)
        self.assertAlmostEqual(probabilities["Depressed"], 0.7)

    def test_missing_feature_is_rejected(self):
        answers = valid_answers()
        answers.pop("Age")

        with self.assertRaises(HTTPException) as context:
            api.predict_from_form(
                api.ScreeningFormSubmission(answers=answers)
            )

        self.assertEqual(context.exception.status_code, 422)
        self.assertEqual(context.exception.detail["missing_features"], ["Age"])

    def test_invalid_feature_value_is_rejected(self):
        answers = valid_answers()
        answers["Age"] = 4

        with self.assertRaises(HTTPException) as context:
            api.predict_from_form(
                api.ScreeningFormSubmission(answers=answers)
            )

        self.assertEqual(context.exception.status_code, 422)
        self.assertEqual(context.exception.detail["invalid_features"], ["Age"])

    def test_self_harm_response_stops_without_running_model(self):
        answers = valid_answers()
        answers[SYMPTOM_FEATURES[-1]] = 1

        with patch.dict(sys.modules, {"src.predict_transformer": None}):
            result = api.predict_from_form(
                api.ScreeningFormSubmission(answers=answers)
            )

        self.assertEqual(
            result["screening_result"]["status"],
            "safety_interrupted",
        )
        self.assertIn(
            "emergency services",
            result["screening_result"]["message"],
        )


class AzureSpeechTests(unittest.TestCase):
    def test_missing_credentials_reports_service_unavailable(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(HTTPException) as context:
                api.synthesize_voice(api.VoiceSynthesisRequest(text="Hello"))

        self.assertEqual(context.exception.status_code, 503)

    def test_synthesis_uses_server_key_and_escapes_ssml_text(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b"fake-mp3-audio"
        with (
            patch.dict(
                "os.environ",
                {
                    "AZURE_SPEECH_KEY": "test-secret",
                    "AZURE_SPEECH_REGION": "centralindia",
                },
            ),
            patch("app.urlopen", return_value=response) as urlopen,
        ):
            audio = api.synthesize_voice(
                api.VoiceSynthesisRequest(text="Hello <friend> & welcome")
            )

        azure_request = urlopen.call_args.args[0]
        self.assertEqual(
            azure_request.get_header("Ocp-apim-subscription-key"),
            "test-secret",
        )
        self.assertIn(
            "Hello &lt;friend&gt; &amp; welcome",
            azure_request.data.decode("utf-8"),
        )
        self.assertEqual(audio.media_type, "audio/mpeg")
        self.assertEqual(audio.body, b"fake-mp3-audio")


if __name__ == "__main__":
    unittest.main()
