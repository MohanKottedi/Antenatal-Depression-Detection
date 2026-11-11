import unittest
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database import (
    AssistantConversation,
    AssistantMessage,
    Base,
    MoodEntry,
    NotificationDelivery,
    PatientProfile,
    PregnancyProfile,
    PushSubscription,
    User,
    ScreeningResponse,
    ScreeningResult,
    ScreeningSession,
)
from src.maternal_care import URGENT_ASSISTANT_RESPONSE
from src.sneha.feature_schema import SYMPTOM_FEATURES
from src.wellbeing_api import (
    AssistantMessageRequest,
    MeditationCompletionRequest,
    MoodEntryRequest,
    NutritionRequest,
    PregnancyProfileRequest,
    PushSubscriptionKeys,
    PushSubscriptionRequest,
    PushUnsubscribeRequest,
    ai_nutrition_ideas,
    create_assistant_conversation,
    delete_assistant_conversation,
    delete_mood_entry,
    emergency_resources,
    create_push_subscription,
    delete_push_subscription,
    get_pregnancy_profile,
    list_mood_entries,
    mental_health_progress,
    nutrition_guidance,
    read_assistant_conversation,
    save_meditation_session,
    save_mood_entry,
    send_assistant_message,
    update_pregnancy_profile,
)
from src.notification_scheduler import deliver_due_reminders, due_reminders


class WellbeingApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, autoflush=False)()
        self.user = self._create_patient("patient@example.com")
        self.other_user = self._create_patient("other@example.com")

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _create_patient(self, email):
        user = User(
            full_name="Test Patient",
            email=email,
            password_hash="test-hash",
            role="patient",
            account_status="active",
        )
        self.db.add(user)
        self.db.flush()
        profile = PatientProfile(
            user_id=user.id,
            patient_reference=f"TEST-{user.id}",
        )
        self.db.add(profile)
        self.db.commit()
        return user

    def _profile(self, user):
        return self.db.query(PatientProfile).filter_by(user_id=user.id).one()

    def test_pregnancy_dates_are_private_and_confirmed_due_date_is_separate(self):
        result = update_pregnancy_profile(
            PregnancyProfileRequest(
                lmp_date=date(2026, 1, 1),
                confirmed_due_date=date(2026, 10, 20),
                follow_up_reminder_date=date(2026, 12, 1),
                dietary_preference="vegetarian",
            ),
            self.user,
            self.db,
        )
        self.assertEqual(result["tracker"]["estimated_due_date"], "2026-10-08")
        self.assertEqual(result["tracker"]["due_date"], "2026-10-20")

        own_data = get_pregnancy_profile(self.user, self.db)
        other_data = get_pregnancy_profile(self.other_user, self.db)
        self.assertEqual(own_data["profile"]["dietary_preference"], "vegetarian")
        self.assertIsNone(other_data["profile"]["lmp_date"])
        self.assertIsNone(other_data["profile"]["confirmed_due_date"])

    def test_pregnancy_update_validates_dates_and_preference(self):
        with self.assertRaises(HTTPException) as future_lmp:
            update_pregnancy_profile(
                PregnancyProfileRequest(lmp_date=date(2999, 1, 1)),
                self.user,
                self.db,
            )
        self.assertEqual(future_lmp.exception.status_code, 422)

        with self.assertRaises(HTTPException):
            update_pregnancy_profile(
                PregnancyProfileRequest(dietary_preference="unsupported"),
                self.user,
                self.db,
            )

    def test_reminder_settings_validate_local_time_and_timezone(self):
        with self.assertRaises(HTTPException) as invalid_time:
            update_pregnancy_profile(
                PregnancyProfileRequest(reminder_time="25:00"),
                self.user,
                self.db,
            )
        self.assertEqual(invalid_time.exception.status_code, 422)

        with self.assertRaises(HTTPException) as invalid_timezone:
            update_pregnancy_profile(
                PregnancyProfileRequest(reminder_timezone="Not/A_Zone"),
                self.user,
                self.db,
            )
        self.assertEqual(invalid_timezone.exception.status_code, 422)

        saved = update_pregnancy_profile(
            PregnancyProfileRequest(
                reminder_time="15:41",
                reminder_timezone="Asia/Kolkata",
            ),
            self.user,
            self.db,
        )
        self.assertEqual(saved["profile"]["reminder_time"], "15:41")
        self.assertEqual(saved["profile"]["reminder_timezone"], "Asia/Kolkata")

    def test_push_subscriptions_are_owned_by_the_patient(self):
        with patch.dict(
            "os.environ",
            {
                "VAPID_PUBLIC_KEY": "public",
                "VAPID_PRIVATE_KEY": "private.pem",
            },
        ):
            request = PushSubscriptionRequest(
                endpoint="https://push.example.test/subscription/unique",
                keys=PushSubscriptionKeys(
                    p256dh="p" * 40,
                    auth="a" * 20,
                ),
                reminder_time="15:41",
                reminder_timezone="Asia/Kolkata",
            )
            result = create_push_subscription(request, self.user, self.db)
        self.assertIn("enabled", result["message"])
        self.assertEqual(
            self.db.query(PushSubscription).filter_by(user_id=self.user.id).count(),
            1,
        )
        delete_push_subscription(
            PushUnsubscribeRequest(endpoint=request.endpoint),
            self.other_user,
            self.db,
        )
        self.assertEqual(
            self.db.query(PushSubscription).filter_by(user_id=self.user.id).count(),
            1,
        )
        delete_push_subscription(
            PushUnsubscribeRequest(endpoint=request.endpoint),
            self.user,
            self.db,
        )
        self.assertEqual(self.db.query(PushSubscription).count(), 0)

    def test_due_reminders_use_local_time_and_week_boundaries(self):
        pregnancy_start = date(2025, 11, 9)
        profile = PregnancyProfile(
            patient_id=self._profile(self.user).id,
            lmp_date=pregnancy_start,
            follow_up_reminder_date=date(2026, 3, 8),
            reminder_time="15:41",
            reminder_timezone="America/Los_Angeles",
        )
        self.db.add(profile)
        self.db.commit()
        before = datetime(2026, 3, 8, 22, 40, tzinfo=timezone.utc)
        due = datetime(2026, 3, 8, 22, 42, tzinfo=timezone.utc)
        self.assertEqual(due_reminders(profile, before), [])
        reminders = due_reminders(profile, due)
        self.assertIn(("screening:2026-03-08", 0), reminders)
        self.assertIn(
            (f"pregnancy-week:{pregnancy_start.isoformat()}:17", 17),
            reminders,
        )
        profile.screening_reminders = False
        self.assertEqual(
            due_reminders(profile, due),
            [(f"pregnancy-week:{pregnancy_start.isoformat()}:17", 17)],
        )
        profile.weekly_tracker_reminders = False
        self.assertEqual(due_reminders(profile, due), [])

    def test_due_pushes_are_deduplicated_and_saved(self):
        fixed_now = datetime(2026, 1, 1, 15, 42, tzinfo=timezone.utc)
        profile = PregnancyProfile(
            patient_id=self._profile(self.user).id,
            lmp_date=fixed_now.date() - timedelta(days=70),
            follow_up_reminder_date=fixed_now.date(),
            reminder_time="15:41",
            reminder_timezone="UTC",
        )
        self.db.add(profile)
        self.db.add(
            PushSubscription(
                user_id=self.user.id,
                endpoint="https://push.example.test/subscription/dedup",
                endpoint_hash=sha256(
                    b"https://push.example.test/subscription/dedup"
                ).hexdigest(),
                p256dh_key="p" * 40,
                auth_key="a" * 20,
            )
        )
        self.db.commit()
        sent = []

        def fake_send(subscription, payload):
            sent.append(payload)

        self.assertEqual(deliver_due_reminders(self.db, fixed_now, fake_send), 2)
        self.assertEqual(deliver_due_reminders(self.db, fixed_now, fake_send), 0)
        self.assertEqual(len(sent), 2)
        self.assertEqual(
            self.db.query(NotificationDelivery).filter_by(status="sent").count(),
            2,
        )

    def test_failed_push_delivery_is_logged_and_retried_after_backoff(self):
        fixed_now = datetime(2026, 1, 1, 15, 42, tzinfo=timezone.utc)
        profile = PregnancyProfile(
            patient_id=self._profile(self.user).id,
            follow_up_reminder_date=fixed_now.date(),
            reminder_time="15:41",
            reminder_timezone="UTC",
        )
        endpoint = "https://push.example.test/private-endpoint"
        second_endpoint = "https://push.example.test/second-device"
        self.db.add(profile)
        self.db.add_all(
            [
                PushSubscription(
                    user_id=self.user.id,
                    endpoint=endpoint,
                    endpoint_hash=sha256(endpoint.encode()).hexdigest(),
                    p256dh_key="p" * 40,
                    auth_key="a" * 20,
                ),
                PushSubscription(
                    user_id=self.user.id,
                    endpoint=second_endpoint,
                    endpoint_hash=sha256(second_endpoint.encode()).hexdigest(),
                    p256dh_key="p" * 40,
                    auth_key="a" * 20,
                ),
            ]
        )
        self.db.commit()
        delivered_endpoints = []

        def partial_send(subscription, _payload):
            delivered_endpoints.append(subscription.endpoint)
            if subscription.endpoint == endpoint:
                raise RuntimeError("delivery unavailable")

        self.assertEqual(deliver_due_reminders(self.db, fixed_now, partial_send), 1)
        deliveries = self.db.query(NotificationDelivery).all()
        self.assertCountEqual([row.status for row in deliveries], ["failed", "sent"])
        failed_delivery = next(row for row in deliveries if row.status == "failed")
        self.assertNotIn(endpoint, failed_delivery.last_error)
        self.assertEqual(
            deliver_due_reminders(
                self.db,
                fixed_now + timedelta(minutes=4),
                partial_send,
            ),
            0,
        )
        self.assertEqual(
            deliver_due_reminders(
                self.db,
                fixed_now + timedelta(minutes=6),
                lambda subscription, _payload: delivered_endpoints.append(
                    subscription.endpoint
                ),
            ),
            1,
        )
        self.db.refresh(failed_delivery)
        self.assertEqual(failed_delivery.status, "sent")
        self.assertEqual(delivered_endpoints.count(second_endpoint), 1)

    def test_mood_entries_are_owned_by_patient_and_can_be_updated_or_deleted(self):
        first = save_mood_entry(
            MoodEntryRequest(
                mood="calm",
                entry_date=date.today(),
                stress_level=2,
                sleep_hours=7.5,
                journal="Synthetic test journal.",
            ),
            self.user,
            self.db,
        )["entry"]
        second = save_mood_entry(
            MoodEntryRequest(
                mood="anxious",
                entry_date=date.today(),
                stress_level=4,
            ),
            self.user,
            self.db,
        )["entry"]
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["mood"], "anxious")

        own_entries = list_mood_entries(90, self.user, self.db)
        other_entries = list_mood_entries(90, self.other_user, self.db)
        self.assertEqual(len(own_entries["entries"]), 1)
        self.assertEqual(own_entries["entries"][0]["stress_level"], 4)
        self.assertEqual(other_entries["entries"], [])

        with self.assertRaises(HTTPException) as private_entry:
            delete_mood_entry(first["id"], self.other_user, self.db)
        self.assertEqual(private_entry.exception.status_code, 404)
        self.assertEqual(
            self.db.query(MoodEntry).filter_by(id=first["id"]).count(),
            1,
        )
        delete_mood_entry(first["id"], self.user, self.db)
        self.assertEqual(
            self.db.query(MoodEntry).filter_by(id=first["id"]).count(),
            0,
        )

    def test_assistant_conversation_history_is_private_and_deletable(self):
        conversation = create_assistant_conversation(self.user, self.db)["conversation"]
        send_assistant_message(
            conversation["id"],
            AssistantMessageRequest(message="What are general pregnancy food safety tips?"),
            self.user,
            self.db,
        )
        self.assertEqual(
            self.db.query(AssistantMessage).filter_by(
                conversation_id=conversation["id"],
                role="user",
            ).count(),
            1,
        )

        with self.assertRaises(HTTPException) as private_record:
            read_assistant_conversation(
                conversation["id"],
                self.other_user,
                self.db,
            )
        self.assertEqual(private_record.exception.status_code, 404)

        history = read_assistant_conversation(conversation["id"], self.user, self.db)
        self.assertEqual([item["role"] for item in history["messages"]], ["user", "assistant"])
        delete_assistant_conversation(conversation["id"], self.user, self.db)
        self.assertEqual(
            self.db.query(AssistantConversation).filter_by(id=conversation["id"]).count(),
            0,
        )

    def test_assistant_urgent_notice_does_not_call_llm(self):
        conversation = create_assistant_conversation(self.user, self.db)["conversation"]
        with patch("src.wellbeing_api.SnehaLLMClient.chat") as chat:
            result = send_assistant_message(
                conversation["id"],
                AssistantMessageRequest(message="I have heavy bleeding"),
                self.user,
                self.db,
            )
        chat.assert_not_called()
        self.assertTrue(result["urgent_guidance"])
        self.assertEqual(
            result["assistant_message"]["content"],
            URGENT_ASSISTANT_RESPONSE,
        )

    def test_assistant_failure_is_explicit_and_preserves_user_message(self):
        conversation = create_assistant_conversation(self.user, self.db)["conversation"]
        with patch(
            "src.wellbeing_api.SnehaLLMClient.chat",
            side_effect=RuntimeError("Ollama is unavailable."),
        ):
            with self.assertRaises(HTTPException) as failure:
                send_assistant_message(
                    conversation["id"],
                    AssistantMessageRequest(message="A general question."),
                    self.user,
                    self.db,
                )
        self.assertEqual(failure.exception.status_code, 503)
        records = self.db.query(AssistantMessage).filter_by(
            conversation_id=conversation["id"]
        ).all()
        self.assertEqual([item.role for item in records], ["user"])

    def test_mental_health_progress_separates_model_and_mood_data(self):
        save_mood_entry(
            MoodEntryRequest(mood="neutral"),
            self.user,
            self.db,
        )
        result = mental_health_progress(self.user, self.db)
        self.assertEqual(result["screenings"], [])
        self.assertEqual(result["mood_trend"][0]["mood"], "neutral")
        self.assertIn("Mood entries are self-reported", result["notices"][2])

    def test_progress_reports_questionnaire_sum_separately_from_model_probability(self):
        profile = self._profile(self.user)
        screening = ScreeningSession(
            patient_id=profile.id,
            screening_status="completed",
            completion_percentage=100,
        )
        self.db.add(screening)
        self.db.flush()
        self.db.add_all(
            ScreeningResponse(
                screening_session_id=screening.id,
                question_key=feature,
                response_value="2",
            )
            for feature in SYMPTOM_FEATURES
        )
        self.db.add(ScreeningResult(
            screening_session_id=screening.id,
            model_name="Synthetic test model",
            model_version="test",
            predicted_class="Not Depressed",
            risk_score=0.2,
        ))
        self.db.commit()

        data = mental_health_progress(self.user, self.db)
        self.assertEqual(data["screenings"][0]["symptom_item_total"], 18)
        self.assertEqual(data["screenings"][0]["model_probability"], 0.2)
        self.assertEqual(data["screenings"][0]["model_prediction"], "Not Depressed")

    def test_nutrition_prefers_saved_profile_without_sharing_allergies(self):
        self.db.add(PregnancyProfile(
            patient_id=self._profile(self.user).id,
            dietary_preference="vegan",
            food_allergies="peanuts",
        ))
        self.db.commit()
        self.assertEqual(
            nutrition_guidance(self.user, self.db)["dietary_preference"],
            "vegan",
        )

        with patch("src.wellbeing_api.SnehaLLMClient.chat", return_value="General ideas.") as chat:
            result = ai_nutrition_ideas(
                NutritionRequest(allow_local_allergy_context=False),
                self.user,
                self.db,
            )
        self.assertFalse(result["allergies_shared_with_local_model"])
        self.assertNotIn("peanuts", chat.call_args.args[0][1]["content"])

        with patch("src.wellbeing_api.SnehaLLMClient.chat", return_value="General ideas.") as chat:
            result = ai_nutrition_ideas(
                NutritionRequest(allow_local_allergy_context=True),
                self.user,
                self.db,
            )
        self.assertTrue(result["allergies_shared_with_local_model"])
        self.assertIn("peanuts", chat.call_args.args[0][1]["content"])

    def test_relaxation_completion_accepts_only_reviewed_durations(self):
        saved = save_meditation_session(
            MeditationCompletionRequest(
                activity_id="gentle-breathing",
                duration_seconds=120,
            ),
            self.user,
            self.db,
        )
        self.assertEqual(saved["session"]["duration_seconds"], 120)
        with self.assertRaises(HTTPException):
            save_meditation_session(
                MeditationCompletionRequest(
                    activity_id="gentle-breathing",
                    duration_seconds=121,
                ),
                self.user,
                self.db,
            )

    def test_emergency_page_does_not_invent_unconfigured_contact_numbers(self):
        with patch.dict("os.environ", {}, clear=True):
            resources = emergency_resources(self.user)
        self.assertEqual(resources["contacts"], [])
        self.assertIsNone(resources["region"])
        self.assertIn("local emergency services", resources["guidance"])


if __name__ == "__main__":
    unittest.main()
