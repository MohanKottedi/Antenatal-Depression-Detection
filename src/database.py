from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, func
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
    city: Mapped[str | None] = mapped_column(String(120), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user: Mapped[User | None] = relationship(back_populates="patient_profile")
    screening_sessions: Mapped[list["ScreeningSession"]] = relationship(back_populates="patient")


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
