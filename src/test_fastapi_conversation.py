"""Direct endpoint tests for the FastAPI conversation integration."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import app as api
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder


class StubConversationEngine:
    def __init__(self):
        self.responses = 0

    def start(self):
        return {
            "assistant_message": "How old are you?",
            "progress": {
                "collected": 0,
                "total": 25,
                "remaining": 25,
                "percentage": 0,
            },
            "stage": "background",
            "completed": False,
        }

    def respond(self, _message):
        self.responses += 1
        return {
            "assistant_message": "How many weeks pregnant are you?",
            "progress": {
                "collected": 1,
                "total": 25,
                "remaining": 24,
                "percentage": 4,
            },
            "stage": "background",
            "completed": False,
            "screening_stopped": False,
            "answer_options": ["Yes", "No"],
            "answer_hint": None,
        }


class FastAPIConversationTests(unittest.TestCase):
    def setUp(self):
        api.conversation_sessions.clear()

    def tearDown(self):
        api.conversation_sessions.clear()

    def test_root_serves_frontend(self):
        response = api.frontend()
        path = Path(response.path)

        self.assertEqual(path.name, "index.html")
        self.assertEqual(path.parent.name, "frontend")

    def test_start_and_respond_keep_session_state(self):
        engine = StubConversationEngine()
        with patch.object(api, "ConversationEngine", return_value=engine):
            started = api.start_conversation()

        self.assertEqual(len(started["session_id"]), 32)
        self.assertEqual(started["assistant_message"], "How old are you?")
        self.assertEqual(started["progress"]["total"], 25)

        request = api.ConversationMessage(
            session_id=started["session_id"],
            message="26",
        )
        result = api.respond_to_conversation(request)

        self.assertEqual(result["assistant_message"], "How many weeks pregnant are you?")
        self.assertEqual(result["progress"]["collected"], 1)
        self.assertEqual(result["answer_options"], ["Yes", "No"])
        self.assertEqual(engine.responses, 1)

    def test_response_converts_numpy_prediction_metadata_for_json(self):
        response = api._conversation_response(
            "a" * 32,
            {
                "screening_result": {
                    "metadata": {
                        np.int64(1): np.int64(2),
                    },
                    "probability": np.float64(0.75),
                },
            },
        )

        encoded = json.loads(json.dumps(jsonable_encoder(response)))
        self.assertEqual(
            encoded["screening_result"]["metadata"]["1"],
            2,
        )
        self.assertEqual(
            encoded["screening_result"]["probability"],
            0.75,
        )

    def test_unknown_session_returns_not_found(self):
        request = api.ConversationMessage(
            session_id="0" * 32,
            message="Hello",
        )

        with self.assertRaises(HTTPException) as context:
            api.respond_to_conversation(request)

        self.assertEqual(context.exception.status_code, 404)

    def test_inactive_session_is_removed(self):
        engine = StubConversationEngine()
        with patch.object(api, "ConversationEngine", return_value=engine):
            started = api.start_conversation()
        session = api.conversation_sessions[started["session_id"]]
        session.last_activity = 0

        request = api.ConversationMessage(
            session_id=started["session_id"],
            message="Hello",
        )
        with self.assertRaises(HTTPException) as context:
            api.respond_to_conversation(request)

        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
