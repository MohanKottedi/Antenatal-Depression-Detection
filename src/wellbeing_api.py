"""Authenticated APIs for the patient wellbeing dashboard."""

from datetime import date, datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.auth_service import require_role
from src.database import (
    AssistantConversation,
    AssistantMessage,
    AuditLog,
    MeditationSession,
    MoodEntry,
    PatientProfile,
    PregnancyProfile,
    PushSubscription,
    ScreeningResponse,
    ScreeningResult,
    ScreeningSession,
    User,
    get_db,
)
from src.maternal_care import (
    DIETARY_PREFERENCES,
    MEAL_IDEAS,
    MEDITATION_ACTIVITIES,
    MOOD_OPTIONS,
    NUTRITION_GUIDANCE,
    NUTRITION_REFERENCES,
    URGENT_ASSISTANT_RESPONSE,
    pregnancy_snapshot,
    urgent_symptoms,
)
from src.sneha.llm_client import SnehaLLMClient
from src.sneha.safety import may_indicate_self_harm
from src.sneha.feature_schema import SYMPTOM_FEATURES


router = APIRouter(prefix="/api", tags=["patient wellbeing"])


class PregnancyProfileRequest(BaseModel):
    lmp_date: date | None = None
    confirmed_due_date: date | None = None
    follow_up_reminder_date: date | None = None
    reminder_time: str = Field(default="15:41", pattern=r"^\d{2}:\d{2}$")
    reminder_timezone: str | None = Field(default=None, max_length=100)
    weekly_tracker_reminders: bool = True
    screening_reminders: bool = True
    dietary_preference: str | None = Field(default=None, max_length=30)
    food_allergies: str | None = Field(default=None, max_length=1000)


class PushSubscriptionKeys(BaseModel):
    p256dh: str = Field(min_length=20, max_length=255)
    auth: str = Field(min_length=10, max_length=255)


class PushSubscriptionRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=4096)
    keys: PushSubscriptionKeys
    reminder_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    reminder_timezone: str = Field(min_length=1, max_length=100)


class PushUnsubscribeRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=4096)


class MoodEntryRequest(BaseModel):
    mood: str = Field(min_length=1, max_length=30)
    entry_date: date = Field(default_factory=date.today)
    stress_level: int | None = Field(default=None, ge=1, le=5)
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    journal: str | None = Field(default=None, max_length=2000)


class AssistantMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class NutritionRequest(BaseModel):
    dietary_preference: str | None = Field(default=None, max_length=30)
    food_allergies: str | None = Field(default=None, max_length=1000)
    allow_local_allergy_context: bool = False


class MeditationCompletionRequest(BaseModel):
    activity_id: str = Field(min_length=1, max_length=50)
    duration_seconds: int = Field(ge=1, le=3600)


def _patient_profile(db: Session, user: User) -> PatientProfile:
    profile = db.query(PatientProfile).filter(
        PatientProfile.user_id == user.id
    ).first()
    if profile is None:
        profile = PatientProfile(
            user_id=user.id,
            patient_reference=f"SN-{uuid4().hex[:8].upper()}",
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def _serialize_mood(entry: MoodEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "entry_date": entry.entry_date.isoformat(),
        "mood": entry.mood,
        "stress_level": entry.stress_level,
        "sleep_hours": entry.sleep_hours,
        "journal": entry.journal,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
    }


def _serialize_assistant_conversation(
    conversation: AssistantConversation,
) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


def _get_conversation(
    db: Session,
    conversation_id: int,
    patient_id: int,
) -> AssistantConversation:
    conversation = db.query(AssistantConversation).filter(
        AssistantConversation.id == conversation_id,
        AssistantConversation.patient_id == patient_id,
    ).first()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


def _validate_reminder_settings(changes: dict[str, Any]) -> None:
    reminder_time = changes.get("reminder_time")
    if reminder_time is not None:
        hours, minutes = (int(value) for value in reminder_time.split(":"))
        if hours > 23 or minutes > 59:
            raise HTTPException(status_code=422, detail="Choose a valid reminder time.")
    timezone = changes.get("reminder_timezone")
    if timezone:
        try:
            ZoneInfo(timezone)
        except (ValueError, ZoneInfoNotFoundError):
            raise HTTPException(
                status_code=422,
                detail="Choose a valid IANA timezone.",
            ) from None


def _serialize_pregnancy_profile(profile: PregnancyProfile | None) -> dict[str, Any]:
    return {
        "lmp_date": profile.lmp_date.isoformat() if profile and profile.lmp_date else None,
        "confirmed_due_date": (
            profile.confirmed_due_date.isoformat()
            if profile and profile.confirmed_due_date
            else None
        ),
        "follow_up_reminder_date": (
            profile.follow_up_reminder_date.isoformat()
            if profile and profile.follow_up_reminder_date
            else None
        ),
        "reminder_time": profile.reminder_time if profile else "15:41",
        "reminder_timezone": profile.reminder_timezone if profile else None,
        "weekly_tracker_reminders": (
            profile.weekly_tracker_reminders if profile else True
        ),
        "screening_reminders": profile.screening_reminders if profile else True,
        "dietary_preference": profile.dietary_preference if profile else "no_preference",
        "food_allergies": profile.food_allergies if profile else None,
    }


@router.get("/notifications/status")
def notification_status(
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    subscription_count = db.query(PushSubscription).filter(
        PushSubscription.user_id == user.id
    ).count()
    public_key = os.getenv("VAPID_PUBLIC_KEY") or None
    return {
        "push_configured": bool(
            public_key and os.getenv("VAPID_PRIVATE_KEY")
        ),
        "subscribed": subscription_count > 0,
        "subscription_count": subscription_count,
        "public_key": public_key,
        "reminder_time": (
            db.query(PregnancyProfile)
            .filter(PregnancyProfile.patient_id == patient.id)
            .with_entities(PregnancyProfile.reminder_time)
            .scalar()
        ) or "15:41",
    }


@router.post("/notifications/subscription")
def create_push_subscription(
    payload: PushSubscriptionRequest,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    if not os.getenv("VAPID_PUBLIC_KEY") or not os.getenv("VAPID_PRIVATE_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Push notifications are not configured on this server.",
        )
    if not payload.endpoint.startswith("https://"):
        raise HTTPException(status_code=422, detail="Push endpoint must use HTTPS.")
    try:
        ZoneInfo(payload.reminder_timezone)
    except (ValueError, ZoneInfoNotFoundError):
        raise HTTPException(
            status_code=422,
            detail="Choose a valid IANA timezone.",
        ) from None
    hours, minutes = (int(value) for value in payload.reminder_time.split(":"))
    if hours > 23 or minutes > 59:
        raise HTTPException(status_code=422, detail="Choose a valid reminder time.")

    endpoint_hash = sha256(payload.endpoint.encode("utf-8")).hexdigest()
    existing = db.query(PushSubscription).filter(
        PushSubscription.endpoint_hash == endpoint_hash
    ).first()
    if existing is not None and existing.user_id != user.id:
        raise HTTPException(status_code=409, detail="This browser subscription is already in use.")
    if existing is None:
        count = db.query(PushSubscription).filter(
            PushSubscription.user_id == user.id
        ).count()
        if count >= 5:
            raise HTTPException(
                status_code=409,
                detail="Remove a browser subscription before adding another device.",
            )
        existing = PushSubscription(
            user_id=user.id,
            endpoint=payload.endpoint,
            endpoint_hash=endpoint_hash,
            p256dh_key=payload.keys.p256dh,
            auth_key=payload.keys.auth,
        )
        db.add(existing)
    else:
        existing.p256dh_key = payload.keys.p256dh
        existing.auth_key = payload.keys.auth

    patient = _patient_profile(db, user)
    profile = db.query(PregnancyProfile).filter(
        PregnancyProfile.patient_id == patient.id
    ).first()
    if profile is None:
        profile = PregnancyProfile(patient_id=patient.id)
        db.add(profile)
    profile.reminder_time = payload.reminder_time
    profile.reminder_timezone = payload.reminder_timezone
    db.commit()
    return {"message": "Background reminders are enabled for this browser."}


@router.delete("/notifications/subscription")
def delete_push_subscription(
    payload: PushUnsubscribeRequest,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    subscription = db.query(PushSubscription).filter(
        PushSubscription.user_id == user.id,
        PushSubscription.endpoint == payload.endpoint,
    ).first()
    if subscription is not None:
        db.delete(subscription)
        db.commit()
    return {"message": "This browser will no longer receive Sneha reminders."}


@router.get("/pregnancy")
def get_pregnancy_profile(
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    profile = db.query(PregnancyProfile).filter(
        PregnancyProfile.patient_id == patient.id
    ).first()
    lmp_date = profile.lmp_date if profile else None
    confirmed_due_date = profile.confirmed_due_date if profile else None
    return {
        "profile": _serialize_pregnancy_profile(profile),
        "tracker": pregnancy_snapshot(lmp_date, confirmed_due_date),
    }


@router.put("/pregnancy")
def update_pregnancy_profile(
    payload: PregnancyProfileRequest,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one pregnancy or nutrition field.",
        )
    _validate_reminder_settings(changes)
    if changes.get("lmp_date") and changes["lmp_date"] > date.today():
        raise HTTPException(status_code=422, detail="LMP date cannot be in the future.")
    preference = changes.get("dietary_preference")
    if "dietary_preference" in changes and preference not in DIETARY_PREFERENCES:
        raise HTTPException(status_code=422, detail="Choose a supported dietary preference.")
    if "food_allergies" in changes and changes["food_allergies"] is not None:
        changes["food_allergies"] = changes["food_allergies"].strip() or None

    patient = _patient_profile(db, user)
    profile = db.query(PregnancyProfile).filter(
        PregnancyProfile.patient_id == patient.id
    ).first()
    if profile is None:
        profile = PregnancyProfile(patient_id=patient.id)
        db.add(profile)
    for field, value in changes.items():
        setattr(profile, field, value)
    db.add(AuditLog(
        actor_user_id=user.id,
        action="pregnancy_profile_updated",
        resource_type="pregnancy_profile",
        resource_id=str(patient.id),
    ))
    db.commit()
    db.refresh(profile)
    return {
        "message": "Pregnancy and nutrition preferences saved.",
        "profile": _serialize_pregnancy_profile(profile),
        "tracker": pregnancy_snapshot(profile.lmp_date, profile.confirmed_due_date),
    }


@router.get("/mood/entries")
def list_mood_entries(
    days: int = Query(default=90, ge=1, le=365),
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    start_date = date.today() - timedelta(days=days - 1)
    entries = db.query(MoodEntry).filter(
        MoodEntry.patient_id == patient.id,
        MoodEntry.entry_date >= start_date,
        MoodEntry.entry_date <= date.today(),
    ).order_by(MoodEntry.entry_date.desc()).all()
    mood_counts = {mood: 0 for mood in MOOD_OPTIONS}
    for entry in entries:
        mood_counts[entry.mood] = mood_counts.get(entry.mood, 0) + 1
    return {
        "entries": [_serialize_mood(entry) for entry in entries],
        "mood_counts": mood_counts,
        "days": days,
        "notice": (
            "Mood and sleep notes are self-reported wellbeing information, "
            "not a depression diagnosis."
        ),
    }


@router.post("/mood/entries")
def save_mood_entry(
    payload: MoodEntryRequest,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    if payload.mood not in MOOD_OPTIONS:
        raise HTTPException(status_code=422, detail="Choose one of the listed moods.")
    if payload.entry_date > date.today():
        raise HTTPException(status_code=422, detail="Mood entry date cannot be in the future.")
    if payload.entry_date < date.today() - timedelta(days=365):
        raise HTTPException(status_code=422, detail="Mood entries can be recorded up to one year back.")

    patient = _patient_profile(db, user)
    entry = db.query(MoodEntry).filter(
        MoodEntry.patient_id == patient.id,
        MoodEntry.entry_date == payload.entry_date,
    ).first()
    if entry is None:
        entry = MoodEntry(patient_id=patient.id, entry_date=payload.entry_date)
        db.add(entry)
    entry.mood = payload.mood
    entry.stress_level = payload.stress_level
    entry.sleep_hours = payload.sleep_hours
    entry.journal = payload.journal.strip() if payload.journal else None
    db.add(AuditLog(
        actor_user_id=user.id,
        action="mood_entry_saved",
        resource_type="mood_entry",
        resource_id=payload.entry_date.isoformat(),
    ))
    try:
        db.commit()
        db.refresh(entry)
    except SQLAlchemyError:
        db.rollback()
        raise
    return {"message": "Mood entry saved.", "entry": _serialize_mood(entry)}


@router.delete("/mood/entries/{entry_id}")
def delete_mood_entry(
    entry_id: int,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    entry = db.query(MoodEntry).filter(
        MoodEntry.id == entry_id,
        MoodEntry.patient_id == patient.id,
    ).first()
    if entry is None:
        raise HTTPException(status_code=404, detail="Mood entry not found.")
    db.delete(entry)
    db.add(AuditLog(
        actor_user_id=user.id,
        action="mood_entry_deleted",
        resource_type="mood_entry",
        resource_id=str(entry_id),
    ))
    db.commit()
    return {"message": "Mood entry deleted."}


@router.get("/mental-health/progress")
def mental_health_progress(
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    sessions = db.query(ScreeningSession).filter(
        ScreeningSession.patient_id == patient.id
    ).order_by(desc(ScreeningSession.screening_date)).all()
    history = []
    for session in sessions:
        result = db.query(ScreeningResult).filter(
            ScreeningResult.screening_session_id == session.id
        ).order_by(desc(ScreeningResult.created_at)).first()
        responses = db.query(ScreeningResponse).filter(
            ScreeningResponse.screening_session_id == session.id,
            ScreeningResponse.question_key.in_(SYMPTOM_FEATURES),
        ).all()
        symptom_responses = {
            item.question_key: item.response_value
            for item in responses
        }
        symptom_total = None
        if set(symptom_responses) == set(SYMPTOM_FEATURES):
            try:
                symptom_total = sum(
                    float(symptom_responses[feature])
                    for feature in SYMPTOM_FEATURES
                )
            except ValueError:
                symptom_total = None
        history.append({
            "id": session.id,
            "date": session.screening_date.isoformat(),
            "status": session.screening_status,
            "symptom_item_total": symptom_total,
            "symptom_item_count": len(symptom_responses),
            "model_prediction": result.predicted_class if result else None,
            "model_probability": result.risk_score if result else None,
            "model_name": result.model_name if result else None,
        })

    cutoff = date.today() - timedelta(days=89)
    moods = db.query(MoodEntry).filter(
        MoodEntry.patient_id == patient.id,
        MoodEntry.entry_date >= cutoff,
        MoodEntry.entry_date <= date.today(),
    ).order_by(MoodEntry.entry_date.asc()).all()
    pregnancy_profile = db.query(PregnancyProfile).filter(
        PregnancyProfile.patient_id == patient.id
    ).first()
    return {
        "screenings": history,
        "mood_trend": [
            {
                "date": entry.entry_date.isoformat(),
                "mood": entry.mood,
                "stress_level": entry.stress_level,
                "sleep_hours": entry.sleep_hours,
            }
            for entry in moods
        ],
        "screening_count": len(history),
        "follow_up_reminder_date": (
            pregnancy_profile.follow_up_reminder_date.isoformat()
            if pregnancy_profile and pregnancy_profile.follow_up_reminder_date
            else None
        ),
        "notices": [
            (
                "Symptom item totals are simple sums of the nine questionnaire "
                "responses. They are not validated clinical scores."
            ),
            (
                "Model predictions and probabilities are estimates, not "
                "diagnoses. Changes do not confirm a clinical change."
            ),
            "Mood entries are self-reported and do not diagnose depression.",
        ],
    }


@router.post("/assistant/conversations")
def create_assistant_conversation(
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    conversation = AssistantConversation(
        patient_id=patient.id,
        title="Pregnancy support chat",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return {"conversation": _serialize_assistant_conversation(conversation)}


@router.get("/assistant/conversations")
def list_assistant_conversations(
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    conversations = db.query(AssistantConversation).filter(
        AssistantConversation.patient_id == patient.id
    ).order_by(desc(AssistantConversation.updated_at)).limit(50).all()
    return {"conversations": [_serialize_assistant_conversation(item) for item in conversations]}


@router.get("/assistant/conversations/{conversation_id}")
def read_assistant_conversation(
    conversation_id: int,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    conversation = _get_conversation(db, conversation_id, patient.id)
    messages = db.query(AssistantMessage).filter(
        AssistantMessage.conversation_id == conversation.id
    ).order_by(desc(AssistantMessage.created_at)).limit(200).all()
    return {
        "conversation": _serialize_assistant_conversation(conversation),
        "messages": [
            {
                "id": item.id,
                "role": item.role,
                "content": item.content,
                "created_at": item.created_at.isoformat(),
            }
            for item in reversed(messages)
        ],
        "has_older_messages": (
            db.query(AssistantMessage).filter(
                AssistantMessage.conversation_id == conversation.id
            ).count() > len(messages)
        ),
    }


@router.delete("/assistant/conversations/{conversation_id}")
def delete_assistant_conversation(
    conversation_id: int,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    conversation = _get_conversation(db, conversation_id, patient.id)
    db.delete(conversation)
    db.add(AuditLog(
        actor_user_id=user.id,
        action="assistant_conversation_deleted",
        resource_type="assistant_conversation",
        resource_id=str(conversation_id),
    ))
    db.commit()
    return {"message": "Assistant conversation deleted."}


@router.post("/assistant/conversations/{conversation_id}/messages")
def send_assistant_message(
    conversation_id: int,
    payload: AssistantMessageRequest,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    conversation = _get_conversation(db, conversation_id, patient.id)
    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")

    recent = db.query(AssistantMessage).filter(
        AssistantMessage.conversation_id == conversation.id
    ).order_by(desc(AssistantMessage.created_at)).limit(10).all()
    history = [
        {"role": item.role, "content": item.content}
        for item in reversed(recent)
    ]
    user_record = AssistantMessage(
        conversation_id=conversation.id,
        role="user",
        content=user_message,
    )
    db.add(user_record)
    conversation.updated_at = datetime.utcnow()
    if not history:
        conversation.title = user_message[:117] + ("..." if len(user_message) > 120 else "")
    db.commit()
    db.refresh(user_record)

    if may_indicate_self_harm(user_message) or urgent_symptoms(user_message):
        assistant_text = URGENT_ASSISTANT_RESPONSE
    else:
        system_prompt = (
            "You are Sneha, a warm pregnancy education companion, not a doctor. "
            "Give general, cautious information about pregnancy, nutrition, "
            "common symptoms, and emotional wellbeing. Do not diagnose, assess "
            "urgency, prescribe or recommend medication/supplements, or replace "
            "professional care. If you are unsure, say so and recommend asking a "
            "qualified maternity clinician. Do not invent facts or citations. "
            "Trusted references when relevant: NHS pregnancy week by week "
            "(https://www.nhs.uk/pregnancy/week-by-week/), NHS pregnancy diet "
            "(https://www.nhs.uk/pregnancy/keeping-well/have-a-healthy-diet/), "
            "and ACOG due date estimation "
            "(https://www.acog.org/clinical/clinical-guidance/committee-opinion/"
            "articles/2017/05/methods-for-estimating-the-due-date). Name only "
            "these references when relevant; do not claim to have browsed them. "
            "If a message suggests an emergency or "
            "self-harm, tell the user to contact local emergency services or "
            "urgent professional care now. Do not request identifying details. "
            "This is a local Ollama session; never imply that you are a person."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]
        try:
            assistant_text = SnehaLLMClient().chat(messages, temperature=0.3)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The local pregnancy assistant is unavailable. Your message "
                    "was saved; check that Ollama and qwen2.5:7b are running, "
                    "then try again."
                ),
            ) from exc

    assistant_record = AssistantMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_text,
    )
    conversation.updated_at = datetime.utcnow()
    db.add(assistant_record)
    db.commit()
    db.refresh(assistant_record)
    return {
        "user_message": {
            "id": user_record.id,
            "content": user_record.content,
            "created_at": user_record.created_at.isoformat(),
        },
        "assistant_message": {
            "id": assistant_record.id,
            "content": assistant_record.content,
            "created_at": assistant_record.created_at.isoformat(),
        },
        "urgent_guidance": (
            assistant_text == URGENT_ASSISTANT_RESPONSE
        ),
    }


@router.get("/nutrition")
def nutrition_guidance(
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    profile = db.query(PregnancyProfile).filter(
        PregnancyProfile.patient_id == patient.id
    ).first()
    preference = profile.dietary_preference if profile else "no_preference"
    pregnancy = pregnancy_snapshot(
        profile.lmp_date if profile else None,
        profile.confirmed_due_date if profile else None,
    )
    return {
        "dietary_preference": preference,
        "trimester": pregnancy["trimester"],
        "food_allergies_recorded": bool(profile and profile.food_allergies),
        "meal_ideas": MEAL_IDEAS.get(preference, MEAL_IDEAS["no_preference"]),
        "guidance": NUTRITION_GUIDANCE,
        "references": NUTRITION_REFERENCES,
        "notice": (
            "These are general meal ideas, not individualized nutrition care. "
            "They have not been verified against your allergies. Check every "
            "ingredient and food label, and consult your clinician or dietitian "
            "for personalized advice."
        ),
    }


@router.post("/nutrition/ideas")
def ai_nutrition_ideas(
    payload: NutritionRequest,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    profile = db.query(PregnancyProfile).filter(
        PregnancyProfile.patient_id == patient.id
    ).first()
    pregnancy = pregnancy_snapshot(
        profile.lmp_date if profile else None,
        profile.confirmed_due_date if profile else None,
    )
    preference = payload.dietary_preference or (
        profile.dietary_preference if profile else "no_preference"
    )
    if preference not in DIETARY_PREFERENCES:
        raise HTTPException(status_code=422, detail="Choose a supported dietary preference.")

    allergies = payload.food_allergies
    if allergies is None and profile:
        allergies = profile.food_allergies
    local_allergy_context = None
    if allergies and payload.allow_local_allergy_context:
        local_allergy_context = allergies[:1000]

    allowed_ideas = MEAL_IDEAS[preference]
    prompt = (
        "Create a short, general one-day pregnancy meal inspiration using only "
        "the listed meal ideas; do not add ingredients, quantities, nutrients, "
        "supplements, medication, or medical claims. Do not call it a meal plan "
        "or say it is medically personalized. Mention safe food handling and "
        "that all labels/ingredients need checking. If allergy notes are "
        "provided, avoid naming those foods but still say the result is not "
        "allergy-safe and requires label checks. Dietary preference: "
        f"{preference}. Pregnancy trimester estimate for context: "
        f"{pregnancy['trimester'] or 'not recorded'}. Approved ideas: {allowed_ideas}. "
        f"Allergy notes: {local_allergy_context or 'not shared with the model'}."
    )
    try:
        generated = SnehaLLMClient().chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You provide cautious general food inspiration for "
                        "pregnancy. Follow user instructions only when they "
                        "do not conflict with safety. Never diagnose or prescribe."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local AI meal inspiration is unavailable. Try the reviewed meal ideas instead.",
        ) from exc

    return {
        "suggestion": generated,
        "allergies_shared_with_local_model": bool(local_allergy_context),
        "notice": (
            "AI-generated general inspiration is not medical advice or "
            "allergy-safe. Check every ingredient and label. Ask your clinician "
            "or a registered dietitian about individual needs."
        ),
        "references": NUTRITION_REFERENCES,
    }


@router.get("/meditation/activities")
def meditation_activities(
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient = _patient_profile(db, user)
    completed = db.query(MeditationSession).filter(
        MeditationSession.patient_id == patient.id
    ).order_by(desc(MeditationSession.completed_at)).limit(10).all()
    return {
        "activities": [
            {"id": activity_id, **activity}
            for activity_id, activity in MEDITATION_ACTIVITIES.items()
        ],
        "recent_sessions": [
            {
                "activity_id": session.activity_id,
                "duration_seconds": session.duration_seconds,
                "completed_at": session.completed_at.isoformat(),
            }
            for session in completed
        ],
        "notice": (
            "Choose a comfortable position and breathe normally. Stop if you "
            "feel pain, dizziness, breathlessness, or discomfort."
        ),
    }


@router.post("/meditation/sessions")
def save_meditation_session(
    payload: MeditationCompletionRequest,
    user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    activity = MEDITATION_ACTIVITIES.get(payload.activity_id)
    if activity is None or payload.duration_seconds not in activity["durations"]:
        raise HTTPException(status_code=422, detail="Choose a listed activity and duration.")
    patient = _patient_profile(db, user)
    session = MeditationSession(
        patient_id=patient.id,
        activity_id=payload.activity_id,
        duration_seconds=payload.duration_seconds,
    )
    db.add(session)
    db.add(AuditLog(
        actor_user_id=user.id,
        action="meditation_completed",
        resource_type="meditation_session",
        resource_id=payload.activity_id,
    ))
    db.commit()
    db.refresh(session)
    return {
        "message": "Relaxation session recorded.",
        "session": {
            "activity_id": session.activity_id,
            "duration_seconds": session.duration_seconds,
            "completed_at": session.completed_at.isoformat(),
        },
    }


@router.get("/emergency/resources")
def emergency_resources(
    user: User = Depends(require_role("patient")),
):
    import os

    contacts = []
    for key, label in (
        ("MATERNAL_EMERGENCY_NUMBER", "Local emergency services"),
        ("MATERNAL_MATERNITY_CONTACT", "Maternity care contact"),
        ("MATERNAL_MENTAL_HEALTH_CONTACT", "Mental health crisis support"),
    ):
        value = os.getenv(key, "").strip()
        if value:
            contacts.append({"label": label, "contact": value})
    return {
        "region": os.getenv("MATERNAL_SUPPORT_REGION", "").strip() or None,
        "contacts": contacts,
        "guidance": (
            "For severe, rapidly worsening, or worrying symptoms, contact your "
            "maternity clinician or local emergency services now. If you may "
            "harm yourself or cannot stay safe, contact emergency services and "
            "a trusted person who can stay with you. Do not wait for an AI reply."
        ),
        "warning_signs": [
            "Heavy vaginal bleeding or fluid leaking.",
            "Severe headache, vision changes, or a seizure.",
            "Chest pain, trouble breathing, or fainting.",
            "Severe abdominal pain or a significant change in the baby's movement.",
            "Thoughts of self-harm or feeling unable to stay safe.",
        ],
        "references": [
            {
                "label": "CDC: urgent maternal warning signs",
                "url": "https://www.cdc.gov/hearher/maternal-warning-signs/index.html",
            },
            {
                "label": "ACOG: urgent maternal warning signs",
                "url": (
                    "https://www.acog.org/womens-health/infographics/"
                    "urgent-maternal-warning-signs"
                ),
            },
        ],
        "map_search": {
            "label": "Find a nearby maternity hospital",
            "requires_user_action": True,
            "notice": (
                "The map opens only when you choose it. Your browser will ask "
                "permission to share your location with the map provider."
            ),
        },
    }
