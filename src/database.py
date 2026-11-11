from __future__ import annotations

import os
from datetime import date, datetime
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    pass


def _database_url() -> str:
    value = os.getenv("DATABASE_URL")
    if value:
        return value
    return "sqlite:///./sneha.db"


engine = create_engine(
    _database_url(),
    connect_args={"check_same_thread": False} if _database_url().startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(30), default="patient", nullable=False)
    google_subject: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=True,
    )
    account_status: Mapped[str] = mapped_column(String(30), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    patient_profile: Mapped["PatientProfile | None"] = relationship(back_populates="user", uselist=False)


class PatientProfile(Base):
    __tablename__ = "patient_profiles"
    __table_args__ = (UniqueConstraint("user_id", name="uq_patient_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    patient_reference: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gestational_age: Mapped[float | None] = mapped_column(Float, nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(40), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    city: Mapped[str | None] = mapped_column(String(120), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped[User | None] = relationship(back_populates="patient_profile")
    screening_sessions: Mapped[list["ScreeningSession"]] = relationship(back_populates="patient")
    pregnancy_profile: Mapped["PregnancyProfile | None"] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        uselist=False,
    )
    mood_entries: Mapped[list["MoodEntry"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    assistant_conversations: Mapped[list["AssistantConversation"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    meditation_sessions: Mapped[list["MeditationSession"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class PregnancyProfile(Base):
    __tablename__ = "pregnancy_profiles"
    __table_args__ = (UniqueConstraint("patient_id", name="uq_pregnancy_profile_patient"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lmp_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    confirmed_due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    follow_up_reminder_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    reminder_time: Mapped[str] = mapped_column(
        String(5),
        default="15:41",
        nullable=False,
    )
    reminder_timezone: Mapped[str | None] = mapped_column(String(100), nullable=True)
    weekly_tracker_reminders: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    screening_reminders: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    dietary_preference: Mapped[str] = mapped_column(
        String(30),
        default="no_preference",
        nullable=False,
    )
    food_allergies: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    patient: Mapped[PatientProfile] = relationship(back_populates="pregnancy_profile")


class MoodEntry(Base):
    __tablename__ = "mood_entries"
    __table_args__ = (
        UniqueConstraint("patient_id", "entry_date", name="uq_mood_patient_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entry_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    mood: Mapped[str] = mapped_column(String(30), nullable=False)
    stress_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    journal: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    patient: Mapped[PatientProfile] = relationship(back_populates="mood_entries")


class AssistantConversation(Base):
    __tablename__ = "assistant_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    patient: Mapped[PatientProfile] = relationship(back_populates="assistant_conversations")
    messages: Mapped[list["AssistantMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AssistantMessage.created_at",
    )


class AssistantMessage(Base):
    __tablename__ = "assistant_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("assistant_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    conversation: Mapped[AssistantConversation] = relationship(back_populates="messages")


class MeditationSession(Base):
    __tablename__ = "meditation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    activity_id: Mapped[str] = mapped_column(String(50), nullable=False)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    patient: Mapped[PatientProfile] = relationship(back_populates="meditation_sessions")


class PushSubscription(Base):
    __tablename__ = "push_subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    endpoint: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    endpoint_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    p256dh_key: Mapped[str] = mapped_column(String(255), nullable=False)
    auth_key: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class NotificationDelivery(Base):
    __tablename__ = "notification_deliveries"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "reminder_key",
            name="uq_notification_user_reminder",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reminder_key: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    last_attempt_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(500), nullable=True)


class DoctorProfile(Base):
    __tablename__ = "doctor_profiles"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_doctor_profile_user"),
        UniqueConstraint(
            "medical_registration_number",
            name="uq_doctor_medical_registration_number",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    medical_registration_number: Mapped[str] = mapped_column(String(120), nullable=False)
    qualification: Mapped[str] = mapped_column(String(255), nullable=False)
    specialization: Mapped[str] = mapped_column(String(255), nullable=False)
    hospital: Mapped[str] = mapped_column(String(255), nullable=False)
    years_of_experience: Mapped[int] = mapped_column(Integer, nullable=False)
    contact_phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    verified_by_user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DoctorPatient(Base):
    __tablename__ = "doctor_patients"
    __table_args__ = (
        UniqueConstraint(
            "doctor_id",
            "patient_id",
            name="uq_doctor_patient_relationship",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    doctor_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("doctor_profiles.id"),
        nullable=False,
        index=True,
    )
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
        index=True,
    )
    consent_status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
        nullable=False,
    )
    relationship_status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
    )
    requested_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    consented_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ScreeningSession(Base):
    __tablename__ = "screening_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("patient_profiles.id"), nullable=True)
    screening_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    screening_status: Mapped[str] = mapped_column(String(30), default="in_progress", nullable=False)
    completion_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    patient: Mapped[PatientProfile | None] = relationship(back_populates="screening_sessions")
    responses: Mapped[list["ScreeningResponse"]] = relationship(back_populates="screening_session")
    results: Mapped[list["ScreeningResult"]] = relationship(back_populates="screening_session")


class ScreeningResponse(Base):
    __tablename__ = "screening_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    screening_session_id: Mapped[int] = mapped_column(Integer, ForeignKey("screening_sessions.id"), nullable=False)
    question_key: Mapped[str] = mapped_column(String(255), nullable=False)
    response_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    screening_session: Mapped[ScreeningSession] = relationship(back_populates="responses")


class ScreeningResult(Base):
    __tablename__ = "screening_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    screening_session_id: Mapped[int] = mapped_column(Integer, ForeignKey("screening_sessions.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=True)
    predicted_class: Mapped[str] = mapped_column(String(80), nullable=False)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    screening_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    screening_session: Mapped[ScreeningSession] = relationship(back_populates="results")


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    doctor_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("doctor_profiles.id"),
        nullable=False,
    )
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )
    screening_session_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("screening_sessions.id"),
        nullable=True,
    )
    note: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    doctor_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("doctor_profiles.id"),
        nullable=False,
    )
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )
    screening_session_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("screening_sessions.id"),
        nullable=True,
    )
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    actor_user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_audit_entry(db: Session, action: str, resource_type: str | None, resource_id: str | None, actor_user_id: int | None = None) -> None:
    entry = AuditLog(
        actor_user_id=actor_user_id,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
    )
    db.add(entry)
    db.commit()


def ensure_admin_account(db: Session) -> User | None:
    from src.auth_service import hash_password

    email = os.getenv("ADMIN_EMAIL")
    if not email:
        return None

    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if user is not None:
        return user

    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        return None

    admin = User(
        full_name="Sneha Admin",
        email=email.lower().strip(),
        password_hash=hash_password(password),
        role="admin",
        account_status="active",
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    create_audit_entry(db, "admin_created", "user", str(admin.id), actor_user_id=admin.id)
    return admin
