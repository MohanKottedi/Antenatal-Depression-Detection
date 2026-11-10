import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as HttpRequest, urlopen
from uuid import uuid4
from xml.sax.saxutils import escape
from _thread import LockType

import joblib
import numpy as np
import pandas as pd
import torch
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse, Response as FastAPIResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, EmailStr, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

base_dir = Path(__file__).resolve().parent
load_dotenv(base_dir / ".env")

from model import FTTransformer, NUMERIC_FEATURE_NAMES, RAW_FEATURE_NAMES
from src.auth_service import COOKIE_NAME, SECRET_KEY, create_access_token, decode_access_token, get_current_user, hash_password, require_role, verify_password
from src.database import AuditLog, ClinicalNote, DoctorPatient, DoctorProfile, PatientProfile, Recommendation, ScreeningResponse, ScreeningResult, ScreeningSession, SessionLocal, User, create_audit_entry, ensure_admin_account, get_db, init_db
from src.sneha.conversation_engine import ConversationEngine
from src.sneha.conversation_manager import QUESTION_TEMPLATES
from src.sneha.feature_extractor import CATEGORY_VALUES, validate_update
from src.sneha.feature_schema import (
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA,
    NUMERIC_FEATURES,
    SYMPTOM_FEATURES,
    SYMPTOM_SCALE,
)

app = FastAPI(title="Antenatal Depression Screening API")
cookie_secure = os.getenv("COOKIE_SECURE", "false").strip().lower() == "true"
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=600,
    same_site="lax",
    https_only=cookie_secure,
)
frontend_dir = base_dir / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

ft_model_path = base_dir / "models" / "ft_transformer_antenatal_model.pkl"
legacy_model_path = base_dir / "models" / "screening_pipeline.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model = None
ft_checkpoint = None
pipeline = None
model_kind = "unavailable"
model_load_error = None


@dataclass
class ConversationSession:
    engine: ConversationEngine
    lock: LockType
    last_activity: float


class ConversationMessage(BaseModel):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    message: str = Field(max_length=4000)


class ScreeningFormSubmission(BaseModel):
    answers: dict[str, Any]


class VoiceSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


conversation_sessions: dict[str, ConversationSession] = {}
conversation_sessions_lock = Lock()
SESSION_IDLE_TIMEOUT_SECONDS = 1800


class FeatureVector(BaseModel):
    age: int = 25
    gestational_age_weeks: int = 20
    first_pregnancy: bool = True
    number_of_children: int = 0
    sleep_quality: str = "average"
    stress_level: str = "medium"
    sadness_frequency: str = "sometimes"


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    confirm_password: str = Field(min_length=8, max_length=72)
    role: str = "patient"
    consent: bool = False
    phone: str | None = Field(default=None, max_length=40)
    date_of_birth: date | None = None
    gender: str | None = Field(default=None, max_length=40)
    address: str | None = Field(default=None, max_length=1000)
    medical_registration_number: str | None = Field(default=None, max_length=120)
    qualification: str | None = Field(default=None, max_length=255)
    specialization: str | None = Field(default=None, max_length=255)
    hospital: str | None = Field(default=None, max_length=255)
    years_of_experience: int | None = Field(default=None, ge=0, le=80)

    model_config = ConfigDict(str_strip_whitespace=True)

    def validate_password_match(self) -> None:
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        password_size = len(self.password.encode("utf-8"))
        if password_size < 8:
            raise ValueError("Password must be at least 8 bytes long.")
        if password_size > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes.")
        if self.date_of_birth and self.date_of_birth > date.today():
            raise ValueError("Date of birth cannot be in the future.")
        if self.role not in {"patient", "doctor"}:
            raise ValueError("Role must be patient or doctor.")
        if self.role == "doctor":
            required_fields = {
                "medical registration number": self.medical_registration_number,
                "qualification": self.qualification,
                "specialization": self.specialization,
                "hospital or clinic": self.hospital,
            }
            missing = [label for label, value in required_fields.items() if not value or not value.strip()]
            if missing:
                raise ValueError(f"Doctor registration requires: {', '.join(missing)}.")
            if self.years_of_experience is None or self.years_of_experience < 0:
                raise ValueError("Years of experience must be a non-negative number.")
        elif self.years_of_experience is not None:
            raise ValueError("Years of experience is only valid for doctor registration.")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class DoctorVerificationRequest(BaseModel):
    verification_status: str

    def validate_status(self) -> None:
        if self.verification_status not in {"verified", "rejected"}:
            raise ValueError("Verification status must be verified or rejected.")


class ClinicalNoteRequest(BaseModel):
    note: str = Field(min_length=1, max_length=5000)
    screening_session_id: int | None = None


class RecommendationRequest(BaseModel):
    recommendation: str = Field(min_length=1, max_length=5000)
    screening_session_id: int | None = None


class PatientProfileUpdate(BaseModel):
    age: int | None = Field(default=None, ge=10, le=60)
    gestational_age: float | None = Field(default=None, ge=0, le=45)
    contact_phone: str | None = Field(default=None, max_length=40)
    date_of_birth: date | None = None
    gender: str | None = Field(default=None, max_length=40)
    address: str | None = Field(default=None, max_length=1000)
    city: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=5000)


class ScreeningSaveRequest(BaseModel):
    answers: dict[str, Any]
    screening_status: str = "completed"
    completion_percentage: float = 100.0
    model_name: str = "Transformer"
    model_version: str = "current"
    predicted_class: str | None = None
    risk_score: float | None = None
    screening_summary: str | None = None


@app.on_event("startup")
def startup_event() -> None:
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        secret_key = os.getenv("SECRET_KEY", "")
        if not secret_key or secret_key == "change-me-in-production":
            raise RuntimeError(
                "Set a strong SECRET_KEY before starting in production."
            )
    init_db()
    db = SessionLocal()
    try:
        ensure_admin_account(db)
    finally:
        db.close()


def frontend() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


def _authenticated_destination(request: Request) -> str | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except HTTPException:
        return None

    user_id = payload.get("user_id")
    if user_id is None:
        return None
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None or user.account_status != "active":
            return None
        if user.role == "admin":
            return "/admin"
        if user.role == "doctor":
            return "/doctor"
        if user.role == "patient":
            return "/patient"
        return None
    finally:
        db.close()


@app.get("/", include_in_schema=False, response_model=None)
def frontend_root(request: Request):
    destination = _authenticated_destination(request)
    return RedirectResponse(url=destination or "/auth/login", status_code=302)


@app.get("/screening", include_in_schema=False)
def screening_page(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return RedirectResponse(url="/auth/login", status_code=302)
    try:
        payload = decode_access_token(token)
    except HTTPException:
        return RedirectResponse(url="/auth/login", status_code=302)
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == payload.get("user_id")).first()
        if user is None or user.account_status != "active":
            return RedirectResponse(url="/auth/login", status_code=302)
        if user.role != "patient":
            destination = "/admin" if user.role == "admin" else "/doctor"
            return RedirectResponse(url=destination, status_code=302)
    finally:
        db.close()
    return frontend()


@app.get("/auth/login", include_in_schema=False)
def auth_login(request: Request):
    destination = _authenticated_destination(request)
    if destination:
        return RedirectResponse(url=destination, status_code=302)
    return FileResponse(frontend_dir / "login.html")


def google_oauth_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_CLIENT_ID")
        and os.getenv("GOOGLE_CLIENT_SECRET")
    )


@app.get("/api/auth/google/status")
def google_auth_status():
    return {"configured": google_oauth_configured()}


@app.get("/auth/google")
async def google_login(request: Request):
    if not google_oauth_configured():
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured on this server.",
        )
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or str(
        request.url_for("google_callback")
    )
    return await oauth.google.authorize_redirect(request, redirect_uri)


def _set_auth_cookie(response: Response, user: User) -> None:
    token = create_access_token(user.email, user.id, user.role)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=cookie_secure,
        max_age=60 * 60 * 12,
    )


@app.get("/auth/google/callback")
async def google_callback(request: Request):
    if not google_oauth_configured():
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured on this server.",
        )

    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError:
        return RedirectResponse(
            url="/auth/login?google_error=provider_error",
            status_code=302,
        )
    google_user = token.get("userinfo")
    if not google_user or google_user.get("email_verified") is not True:
        return RedirectResponse(
            url="/auth/login?google_error=email_not_verified",
            status_code=302,
        )

    email = google_user.get("email")
    subject = google_user.get("sub")
    if not email or not subject:
        return RedirectResponse(
            url="/auth/login?google_error=invalid_identity",
            status_code=302,
        )

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.google_subject == subject).first()
        if user is None:
            user = db.query(User).filter(
                User.email == email.lower().strip()
            ).first()
        if user is None:
            return RedirectResponse(
                url="/auth/login?google_error=account_not_found",
                status_code=302,
            )
        if user.google_subject and user.google_subject != subject:
            return RedirectResponse(
                url="/auth/login?google_error=google_account_mismatch",
                status_code=302,
            )
        if user.account_status != "active":
            error = (
                "doctor_pending"
                if user.role == "doctor"
                and user.account_status == "pending_verification"
                else "account_inactive"
            )
            return RedirectResponse(
                url=f"/auth/login?google_error={error}",
                status_code=302,
            )
        if user.role == "doctor":
            doctor_profile = db.query(DoctorProfile).filter(
                DoctorProfile.user_id == user.id
            ).first()
            if (
                doctor_profile is None
                or doctor_profile.verification_status != "verified"
            ):
                return RedirectResponse(
                    url="/auth/login?google_error=doctor_pending",
                    status_code=302,
                )

        user.google_subject = subject
        user.last_login_at = datetime.utcnow()
        db.add(AuditLog(
            actor_user_id=user.id,
            action="user_login_google",
            resource_type="user",
            resource_id=str(user.id),
        ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return RedirectResponse(
                url="/auth/login?google_error=google_account_mismatch",
                status_code=302,
            )
        destination = {
            "admin": "/admin",
            "doctor": "/doctor",
            "patient": "/patient",
        }.get(user.role, "/auth/login")
        response = RedirectResponse(url=destination, status_code=302)
        _set_auth_cookie(response, user)
        return response
    finally:
        db.close()


@app.get("/auth/register", include_in_schema=False)
def auth_register() -> FileResponse:
    return FileResponse(frontend_dir / "register.html")


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(frontend_dir / "admin.html")


@app.get("/patient", include_in_schema=False)
def patient_page() -> FileResponse:
    return FileResponse(frontend_dir / "patient.html")


@app.get("/doctor", include_in_schema=False)
def doctor_page() -> FileResponse:
    return FileResponse(frontend_dir / "doctor.html")


try:
    if ft_model_path.is_file():
        ft_checkpoint = joblib.load(ft_model_path)
        if ft_checkpoint.get("feature_names_raw") != RAW_FEATURE_NAMES:
            raise ValueError("Checkpoint feature names do not match the API schema.")
        if ft_checkpoint.get("classes") != ["Not Depressed", "Depressed"]:
            raise ValueError("Checkpoint classes do not match the API class mapping.")
        ft_model = FTTransformer(ft_checkpoint["model_config"])
        ft_model.load_state_dict(ft_checkpoint["model_state_dict"], strict=True)
        ft_model.to(device).eval()
        model_kind = "ft_transformer"
        print(f"FT-Transformer loaded from {ft_model_path} on {device}.")
    elif legacy_model_path.is_file():
        pipeline = joblib.load(legacy_model_path)
        model_kind = "legacy_pipeline"
        print(f"Legacy screening pipeline loaded from {legacy_model_path}.")
    else:
        model_load_error = (
            f"No model found. Train with train_model.py or provide {legacy_model_path}."
        )
        print(f"Warning: {model_load_error}")
except Exception as exc:
    model_load_error = f"Could not load model: {exc}"
    print(f"Error: {model_load_error}")


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            _to_json_compatible(key): _to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value


def _conversation_response(session_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return _to_json_compatible({
        "session_id": session_id,
        "saved_session_id": result.get("saved_session_id"),
        "assistant_message": result.get("assistant_message"),
        "progress": result.get("progress"),
        "stage": result.get("stage"),
        "completed": result.get("completed", False),
        "screening_stopped": result.get("screening_stopped", False),
        "safety_follow_up": result.get("safety_follow_up", False),
        "screening_result": result.get("screening_result"),
        "answer_options": result.get("answer_options"),
        "answer_hint": result.get("answer_hint"),
    })


def _get_conversation_session(session_id: str) -> ConversationSession:
    _expire_conversation_sessions()
    with conversation_sessions_lock:
        session = conversation_sessions.get(session_id)
        if session is not None:
            session.last_activity = monotonic()

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation session was not found or has ended.",
        )

    return session


def _expire_conversation_sessions() -> None:
    expired_before = monotonic() - SESSION_IDLE_TIMEOUT_SECONDS
    with conversation_sessions_lock:
        expired_ids = [
            session_id
            for session_id, session in conversation_sessions.items()
            if session.last_activity < expired_before
        ]
        for session_id in expired_ids:
            conversation_sessions.pop(session_id)


@app.post("/api/voice/synthesize")
def synthesize_voice(request: VoiceSynthesisRequest):
    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    speech_region = os.environ.get("AZURE_SPEECH_REGION")
    if not speech_key or not speech_region:
        raise HTTPException(
            status_code=503,
            detail=(
                "Azure Speech is not configured. Set AZURE_SPEECH_KEY and "
                "AZURE_SPEECH_REGION on the server."
            ),
        )
    if not re.fullmatch(r"[a-z0-9-]+", speech_region, flags=re.IGNORECASE):
        raise HTTPException(
            status_code=500,
            detail="AZURE_SPEECH_REGION has an invalid format.",
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    ssml = (
        '<speak version="1.0" xml:lang="en-IN">'
        '<voice name="en-IN-NeerjaNeural">'
        f"{escape(text)}"
        "</voice></speak>"
    )
    azure_request = HttpRequest(
        f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1",
        data=ssml.encode("utf-8"),
        headers={
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
            "Ocp-Apim-Subscription-Key": speech_key,
            "User-Agent": "SnehaAntenatalScreening",
        },
        method="POST",
    )
    try:
        with urlopen(azure_request, timeout=20) as response:
            audio = response.read()
    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Azure Speech returned HTTP {exc.code}. Check the server's speech resource configuration.",
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Azure Speech. Check the server's network connection.",
        ) from exc

    if not audio:
        raise HTTPException(
            status_code=502,
            detail="Azure Speech returned an empty audio response.",
        )

    return FastAPIResponse(
        content=audio,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/form/schema")
def screening_form_schema():
    fields = []
    for feature, schema in FEATURE_SCHEMA.items():
        field = {
            "name": feature,
            "label": QUESTION_TEMPLATES[feature],
            "group": schema["group"],
            "type": schema["type"],
        }
        if feature in CATEGORICAL_FEATURES:
            field["options"] = [
                option
                for option in CATEGORY_VALUES[feature]
                if option != "__MISSING__"
            ]
        elif feature in SYMPTOM_FEATURES:
            field["options"] = [
                {"value": value, "label": label}
                for value, label in SYMPTOM_SCALE.items()
            ]
        elif feature in NUMERIC_FEATURES:
            if feature == "Age":
                field.update({"min": 10, "max": 60, "step": 1})
            elif feature == "Gestational Age":
                field.update({"min": 0, "max": 45, "step": "any"})
            else:
                field.update({"min": 0, "step": 1})
        fields.append(field)

    return {"fields": fields}


@app.post("/api/patient/assessments")
@app.post("/api/form/predict")
def predict_from_form(
    request: ScreeningFormSubmission,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    answers = request.answers
    missing = [feature for feature in FEATURE_SCHEMA if feature not in answers]
    unexpected = [feature for feature in answers if feature not in FEATURE_SCHEMA]
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    validated_answers = {}
    invalid_features = []
    for feature in FEATURE_SCHEMA:
        value = answers[feature]
        if not validate_update({"feature": feature, "value": value}):
            invalid_features.append(feature)
            continue
        if feature in NUMERIC_FEATURES or feature in SYMPTOM_FEATURES:
            validated_answers[feature] = float(value)
        else:
            validated_answers[feature] = value

    if invalid_features:
        raise HTTPException(
            status_code=422,
            detail={"invalid_features": invalid_features},
        )

    self_harm_feature = SYMPTOM_FEATURES[-1]
    if validated_answers[self_harm_feature] > 0:
        return {
            "screening_result": {
                "status": "safety_interrupted",
                "message": (
                    "Thank you for sharing that. Are you in danger right now, or "
                    "do you feel you might act on these thoughts? If you may act "
                    "on them or cannot stay safe, contact local emergency services "
                    "now and reach out to someone trusted who can stay with you. "
                    "This screening has stopped and will not produce a model result."
                ),
            }
        }

    progress = {
        "collected": len(FEATURE_SCHEMA),
        "total": len(FEATURE_SCHEMA),
        "remaining": 0,
        "percentage": 100.0,
    }
    try:
        from src.predict_transformer import (
            get_model_info,
            predict_antenatal_depression,
        )

        prediction = predict_antenatal_depression(validated_answers)
        positive_probability = float(prediction["probability"])
        screening_result = {
            "status": "success",
            "model_info": get_model_info(),
            "prediction": prediction,
            "class_probabilities": {
                "Not Depressed": 1.0 - positive_probability,
                "Depressed": positive_probability,
            },
            "progress": progress,
            "completed": True,
        }
    except (FileNotFoundError, ImportError, OSError) as exc:
        screening_result = {
            "status": "unavailable",
            "message": str(exc),
            "prediction": None,
            "class_probabilities": None,
            "progress": progress,
            "completed": True,
        }
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        screening_result = {
            "status": "invalid_features",
            "message": str(exc),
            "prediction": None,
            "class_probabilities": None,
            "progress": progress,
            "completed": True,
        }

    response = {"screening_result": _to_json_compatible(screening_result)}
    if isinstance(current_user, User):
        prediction = screening_result.get("prediction")
        prediction_data = prediction if isinstance(prediction, dict) else {}
        prediction_info = screening_result.get("model_info") or {}
        predicted_class = (
            prediction_data.get("label")
            if screening_result.get("status") == "success"
            else None
        )
        probability = prediction_data.get("probability")
        risk_score = (
            float(probability)
            if predicted_class and probability is not None
            else None
        )
        status = (
            "safety_interrupted"
            if screening_result.get("status") == "safety_interrupted"
            else "completed"
            if screening_result.get("status") == "success"
            else "model_unavailable"
        )
        response["saved_session_id"] = persist_patient_assessment(
            db=db,
            current_user=current_user,
            answers=validated_answers,
            screening_status=status,
            predicted_class=predicted_class,
            risk_score=risk_score,
            model_name=str(prediction_info.get("model") or "Transformer"),
            model_version=str(prediction_info.get("version") or "current"),
        )
    return response


@app.post("/api/conversation/start")
def start_conversation(current_user: User = Depends(require_role("patient"))):
    _expire_conversation_sessions()
    session_id = uuid4().hex
    session = ConversationSession(
        engine=ConversationEngine(),
        lock=Lock(),
        last_activity=monotonic(),
    )

    try:
        result = session.engine.start()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not start the conversation: {exc}",
        ) from exc

    with conversation_sessions_lock:
        conversation_sessions[session_id] = session

    return _conversation_response(session_id, result)


@app.post("/api/conversation/respond")
def respond_to_conversation(
    request: ConversationMessage,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    session = _get_conversation_session(request.session_id)

    try:
        with session.lock:
            result = session.engine.respond(request.message)
            session.last_activity = monotonic()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not process the response: {exc}",
        ) from exc

    response = _conversation_response(request.session_id, result)
    if isinstance(current_user, User) and (
        result.get("completed")
        or (
            result.get("screening_stopped")
            and not result.get("safety_follow_up")
        )
    ):
        answers = session.engine.manager.get_collected_features()
        screening_result = result.get("screening_result") or {}
        prediction = screening_result.get("prediction") or {}
        model_info = screening_result.get("model_info") or {}
        predicted_class = (
            prediction.get("label")
            if screening_result.get("status") == "success"
            else None
        )
        risk_score = (
            float(prediction["probability"])
            if predicted_class and prediction.get("probability") is not None
            else None
        )
        screening_status = (
            "safety_interrupted"
            if result.get("screening_stopped")
            else "completed"
            if predicted_class
            else "model_unavailable"
        )
        response["saved_session_id"] = persist_patient_assessment(
            db=db,
            current_user=current_user,
            answers=answers,
            screening_status=screening_status,
            predicted_class=predicted_class,
            risk_score=risk_score,
            model_name=str(model_info.get("model") or "Transformer"),
            model_version=str(model_info.get("version") or "current"),
        )
    if result.get("completed") or (
        result.get("screening_stopped")
        and not result.get("safety_follow_up")
    ):
        with conversation_sessions_lock:
            conversation_sessions.pop(request.session_id, None)

    return response


@app.get("/health")
def health():
    return {
        "status": "ok" if model_kind != "unavailable" else "model_unavailable",
        "model": model_kind,
        "device": str(device),
        "error": model_load_error,
    }


@app.post("/predict")
def predict(features: dict[str, Any]):
    if model_kind == "unavailable":
        raise HTTPException(status_code=503, detail=model_load_error)

    if ft_model is not None and ft_checkpoint is not None:
        missing = [name for name in RAW_FEATURE_NAMES if name not in features]
        extra = [name for name in features if name not in RAW_FEATURE_NAMES]
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing, "unexpected_features": extra},
            )

        row = {name: features[name] for name in RAW_FEATURE_NAMES}
        for name in NUMERIC_FEATURE_NAMES:
            try:
                row[name] = pd.to_numeric(row[name], errors="raise")
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be numeric.",
                ) from None
        frame = pd.DataFrame([row], columns=RAW_FEATURE_NAMES)
        try:
            transformed = ft_checkpoint["preprocessor"].transform(frame)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            tensor = torch.as_tensor(
                np.asarray(transformed, dtype=np.float32),
                device=device,
            )
            with torch.inference_mode():
                probabilities = torch.softmax(ft_model(tensor), dim=1)[0]
            if probabilities.shape != (2,) or not torch.isfinite(probabilities).all():
                raise ValueError("Model returned invalid class probabilities.")
            risk_proba = float(probabilities[1].item())
            risk_class = int(
                risk_proba >= float(ft_checkpoint["decision_threshold"])
            )
        except HTTPException:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="The legacy screening pipeline is unavailable.",
            )
        try:
            legacy_features = FeatureVector(**features)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        frame = pd.DataFrame([legacy_features.model_dump()])
        risk_class = int(pipeline.predict(frame)[0])
        risk_proba = float(pipeline.predict_proba(frame)[0][1])

    return {
        "risk_class": risk_class,
        "risk_score": risk_proba,
    }


def serialize_user(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "account_status": user.account_status,
    }


def serialize_doctor_profile(profile: DoctorProfile) -> dict[str, Any]:
    return {
        "medical_registration_number": profile.medical_registration_number,
        "qualification": profile.qualification,
        "specialization": profile.specialization,
        "hospital": profile.hospital,
        "years_of_experience": profile.years_of_experience,
        "verification_status": profile.verification_status,
    }


@app.get("/api/admin/doctors")
def list_doctors_for_verification(
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(User, DoctorProfile)
        .join(DoctorProfile, DoctorProfile.user_id == User.id)
        .order_by(DoctorProfile.created_at.desc())
        .all()
    )
    return {
        "doctors": [
            {
                "user": serialize_user(user),
                "profile": serialize_doctor_profile(profile),
                "created_at": profile.created_at.isoformat(),
            }
            for user, profile in rows
        ]
    }


@app.post("/api/admin/doctors/{doctor_user_id}/verification")
def update_doctor_verification(
    doctor_user_id: int,
    payload: DoctorVerificationRequest,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    try:
        payload.validate_status()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    doctor = (
        db.query(User)
        .filter(User.id == doctor_user_id, User.role == "doctor")
        .first()
    )
    profile = (
        db.query(DoctorProfile)
        .filter(DoctorProfile.user_id == doctor_user_id)
        .first()
    )
    if doctor is None or profile is None:
        raise HTTPException(status_code=404, detail="Doctor account not found.")

    profile.verification_status = payload.verification_status
    profile.verified_at = datetime.utcnow()
    profile.verified_by_user_id = current_user.id
    doctor.account_status = (
        "active" if payload.verification_status == "verified" else "inactive"
    )
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action=f"doctor_{payload.verification_status}",
        resource_type="doctor",
        resource_id=str(doctor_user_id),
    ))
    db.commit()
    return {
        "message": f"Doctor account {payload.verification_status}.",
        "user": serialize_user(doctor),
        "profile": serialize_doctor_profile(profile),
    }


def build_depression_summary(predicted_class: str | None, risk_score: float | None) -> str:
    if predicted_class not in {"Depressed", "Not Depressed"}:
        return "No depression prediction was generated for this assessment."

    probability_text = (
        f" The model-estimated probability for the Depressed class was "
        f"{float(risk_score) * 100:.1f}%."
        if risk_score is not None
        else ""
    )
    return (
        f"The model classified this assessment as {predicted_class}."
        f"{probability_text} This is an unvalidated screening estimate, not a "
        "clinical diagnosis. A qualified clinician should assess the patient "
        "independently."
    )


def persist_patient_assessment(
    *,
    db: Session,
    current_user: User,
    answers: dict[str, Any],
    screening_status: str,
    predicted_class: str | None,
    risk_score: float | None,
    model_name: str,
    model_version: str,
) -> int:
    profile = db.query(PatientProfile).filter(
        PatientProfile.user_id == current_user.id
    ).first()
    if profile is None:
        profile = PatientProfile(
            user_id=current_user.id,
            patient_reference=f"SN-{uuid4().hex[:8].upper()}",
        )
        db.add(profile)
        db.flush()
    if "Age" in answers:
        profile.age = int(float(answers["Age"]))
    if "Gestational Age" in answers:
        profile.gestational_age = float(answers["Gestational Age"])

    now = datetime.utcnow()
    screening = ScreeningSession(
        patient_id=profile.id,
        screening_date=now,
        screening_status=screening_status,
        completion_percentage=100.0,
        completed_at=now,
    )
    db.add(screening)
    db.flush()

    db.add_all(
        ScreeningResponse(
            screening_session_id=screening.id,
            question_key=str(key),
            response_value=str(value),
        )
        for key, value in answers.items()
    )

    if predicted_class is not None:
        db.add(
            ScreeningResult(
                screening_session_id=screening.id,
                model_name=model_name,
                model_version=model_version,
                predicted_class=predicted_class,
                risk_score=risk_score,
                screening_summary=build_depression_summary(
                    predicted_class,
                    risk_score,
                ),
            )
        )

    db.add(
        AuditLog(
            actor_user_id=current_user.id,
            action="screening_saved",
            resource_type="screening_session",
            resource_id=str(screening.id),
        )
    )
    try:
        db.commit()
        return screening.id
    except SQLAlchemyError:
        db.rollback()
        raise


def serialize_screening_session(session: ScreeningSession, db: Session) -> dict[str, Any]:
    latest_result = db.query(ScreeningResult).filter(
        ScreeningResult.screening_session_id == session.id
    ).order_by(ScreeningResult.created_at.desc()).first()
    answers = db.query(ScreeningResponse).filter(
        ScreeningResponse.screening_session_id == session.id
    ).order_by(ScreeningResponse.created_at.asc()).all()
    return {
        "id": session.id,
        "screening_date": session.screening_date.isoformat(),
        "screening_status": session.screening_status,
        "completion_percentage": int(session.completion_percentage),
        "predicted_class": latest_result.predicted_class if latest_result else None,
        "risk_score": latest_result.risk_score if latest_result else None,
        "screening_summary": latest_result.screening_summary if latest_result else None,
        "answers": [
            {
                "question_key": item.question_key,
                "response_value": item.response_value,
                "created_at": item.created_at.isoformat(),
            }
            for item in answers
        ],
    }


@app.post("/api/register")
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    if not payload.consent:
        raise HTTPException(status_code=400, detail="Consent is required to create an account.")
    try:
        payload.validate_password_match()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_email = payload.email.lower().strip()
    if db.query(User).filter(User.email == normalized_email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    role = payload.role
    user = User(
        full_name=payload.full_name.strip(),
        email=normalized_email,
        password_hash=hash_password(payload.password),
        role=role,
        account_status="pending_verification" if role == "doctor" else "active",
    )
    db.add(user)
    db.flush()

    if role == "patient":
        profile = PatientProfile(
            user_id=user.id,
            patient_reference=f"SN-{uuid4().hex[:8].upper()}",
            contact_phone=payload.phone.strip() if payload.phone else None,
            date_of_birth=payload.date_of_birth,
            gender=payload.gender.strip() if payload.gender else None,
            address=payload.address.strip() if payload.address else None,
        )
        db.add(profile)
    else:
        profile = DoctorProfile(
            user_id=user.id,
            medical_registration_number=payload.medical_registration_number,
            qualification=payload.qualification,
            specialization=payload.specialization,
            hospital=payload.hospital,
            years_of_experience=payload.years_of_experience,
            contact_phone=payload.phone.strip() if payload.phone else None,
            verification_status="pending",
        )
        db.add(profile)
    db.add(AuditLog(
        actor_user_id=user.id,
        action="user_registered",
        resource_type="user",
        resource_id=str(user.id),
    ))
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail=(
                "This email or medical registration number is already registered."
            ),
        ) from exc
    return {
        "message": (
            "Registration submitted. Your doctor account requires administrator verification."
            if role == "doctor"
            else "Registration successful."
        ),
        "user": serialize_user(user),
    }


@app.post("/api/login")
def login_user(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    normalized_email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == normalized_email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if user.account_status != "active":
        if user.role == "doctor" and user.account_status == "pending_verification":
            raise HTTPException(
                status_code=403,
                detail="Your doctor account is awaiting administrator verification.",
            )
        raise HTTPException(status_code=403, detail="This account is currently inactive.")

    _set_auth_cookie(response, user)
    user.last_login_at = datetime.utcnow()
    db.commit()
    create_audit_entry(db, "user_login", "user", str(user.id), actor_user_id=user.id)
    redirect_url = {
        "admin": "/admin",
        "doctor": "/doctor",
        "patient": "/patient",
    }.get(user.role, "/")
    return {
        "message": "Login successful.",
        "user": serialize_user(user),
        "redirect_url": redirect_url,
    }


@app.post("/api/logout")
def logout_user(response: Response):
    response.delete_cookie(key=COOKIE_NAME, httponly=True, samesite="lax")
    return {"message": "Logout successful."}


@app.get("/api/me")
def get_current_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    doctor_profile = db.query(DoctorProfile).filter(
        DoctorProfile.user_id == current_user.id
    ).first()
    return {
        "user": serialize_user(current_user),
        "profile": {
            "id": profile.id if profile else None,
            "patient_reference": profile.patient_reference if profile else None,
            "age": profile.age if profile else None,
            "gestational_age": profile.gestational_age if profile else None,
            "city": profile.city if profile else None,
            "contact_phone": profile.contact_phone if profile else None,
            "date_of_birth": profile.date_of_birth.isoformat() if profile and profile.date_of_birth else None,
            "gender": profile.gender if profile else None,
            "address": profile.address if profile else None,
            "notes": profile.notes if profile else None,
        } if profile else None,
        "doctor_profile": serialize_doctor_profile(doctor_profile) if doctor_profile else None,
    }


@app.get("/api/admin/dashboard")
def admin_dashboard(current_user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    total_patients = db.query(User).filter(User.role == "patient").count()
    total_sessions = db.query(ScreeningSession).count()
    completed = db.query(ScreeningSession).filter(ScreeningSession.screening_status == "completed").count()
    incomplete = total_sessions - completed
    sessions_last_7_days = db.query(ScreeningSession).filter(ScreeningSession.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).count()
    average_completion = db.query(ScreeningSession.completion_percentage).filter(ScreeningSession.completion_percentage.is_not(None)).with_entities(__import__("sqlalchemy").func.avg(ScreeningSession.completion_percentage)).scalar() or 0
    activity = db.query(ScreeningSession.created_at.label("label"), __import__("sqlalchemy").func.count(ScreeningSession.id).label("value")).group_by(ScreeningSession.created_at).order_by(ScreeningSession.created_at.asc()).limit(7).all()
    outcome_rows = db.query(ScreeningResult.predicted_class.label("label"), __import__("sqlalchemy").func.count(ScreeningResult.id).label("value")).group_by(ScreeningResult.predicted_class).all()
    recent_activity = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(10).all()

    activity_payload = [{"label": row.label.strftime("%b %d") if hasattr(row.label, "strftime") else str(row.label), "value": int(row.value)} for row in activity]
    outcome_payload = [{"label": row.label, "value": int(row.value)} for row in outcome_rows]
    recent_payload = [{"action": item.action, "resource_type": item.resource_type, "created_at": item.created_at.isoformat()} for item in recent_activity]

    return {
        "user": serialize_user(current_user),
        "metrics": {
            "total_patients": total_patients,
            "total_sessions": total_sessions,
            "completed_sessions": completed,
            "incomplete_sessions": incomplete,
            "sessions_last_7_days": sessions_last_7_days,
            "average_completion_rate": round(float(average_completion), 2),
        },
        "activity": activity_payload,
        "outcomes": outcome_payload,
        "recent_activity": recent_payload,
    }


@app.get("/api/admin/patients")
def list_patients(current_user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    patients = db.query(User).filter(User.role == "patient").all()
    output = []
    for user in patients:
        profile = db.query(PatientProfile).filter(PatientProfile.user_id == user.id).first()
        latest_session = (
            db.query(ScreeningSession)
            .filter(ScreeningSession.patient_id == profile.id)
            .order_by(ScreeningSession.created_at.desc())
            .first() if profile else None
        )
        latest_result = (
            db.query(ScreeningResult)
            .join(ScreeningSession, ScreeningSession.id == ScreeningResult.screening_session_id)
            .filter(ScreeningSession.patient_id == profile.id)
            .order_by(ScreeningResult.created_at.desc())
            .first() if profile else None
        )
        output.append({
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "status": user.account_status,
            "patient_reference": profile.patient_reference if profile else None,
            "age": profile.age if profile else None,
            "gestational_age": profile.gestational_age if profile else None,
            "city": profile.city if profile else None,
            "last_screening_date": latest_session.screening_date.isoformat() if latest_session else None,
            "last_prediction": latest_result.predicted_class if latest_result else None,
            "last_risk_score": latest_result.risk_score if latest_result else None,
        })
    return {"patients": output}


@app.get("/api/admin/patients/{patient_id}/history")
def patient_history_for_admin(patient_id: int, current_user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == patient_id, User.role == "patient").first()
    if not user:
        raise HTTPException(status_code=404, detail="Patient not found.")

    profile = db.query(PatientProfile).filter(PatientProfile.user_id == patient_id).first()
    sessions = (
        db.query(ScreeningSession)
        .filter(ScreeningSession.patient_id == profile.id)
        .order_by(ScreeningSession.created_at.desc())
        .all() if profile else []
    )
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action="admin_viewed_patient_history",
        resource_type="patient_profile",
        resource_id=str(profile.id) if profile else str(patient_id),
    ))
    db.commit()
    return {
        "user": serialize_user(user),
        "profile": {
            "patient_reference": profile.patient_reference if profile else None,
            "age": profile.age if profile else None,
            "gestational_age": profile.gestational_age if profile else None,
            "city": profile.city if profile else None,
            "contact_phone": profile.contact_phone if profile else None,
        },
        "screenings": [serialize_screening_session(session, db) for session in sessions],
    }


@app.delete("/api/admin/clear-data")
def clear_all_screening_data(current_user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    db.query(ClinicalNote).delete()
    db.query(Recommendation).delete()
    db.query(ScreeningResponse).delete()
    db.query(ScreeningResult).delete()
    db.query(ScreeningSession).delete()
    db.commit()
    create_audit_entry(db, "screening_data_cleared", "system", "all", actor_user_id=current_user.id)
    return {"message": "Saved screening data and patient analytics have been cleared."}


def _patient_profile_for_user(db: Session, user: User) -> PatientProfile:
    profile = db.query(PatientProfile).filter(
        PatientProfile.user_id == user.id
    ).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Patient profile not found.")
    return profile


def _doctor_profile_for_user(db: Session, user: User) -> DoctorProfile:
    profile = db.query(DoctorProfile).filter(
        DoctorProfile.user_id == user.id
    ).first()
    if profile is None or profile.verification_status != "verified":
        raise HTTPException(status_code=403, detail="Verified doctor access is required.")
    return profile


def _consented_relationship(
    db: Session,
    doctor: DoctorProfile,
    patient_id: int,
) -> DoctorPatient:
    relationship = db.query(DoctorPatient).filter(
        DoctorPatient.doctor_id == doctor.id,
        DoctorPatient.patient_id == patient_id,
        DoctorPatient.consent_status == "consented",
        DoctorPatient.relationship_status == "active",
    ).first()
    if relationship is None:
        raise HTTPException(status_code=404, detail="Patient record not found or not shared with this doctor.")
    return relationship


def _serialize_clinical_note(note: ClinicalNote) -> dict[str, Any]:
    return {
        "id": note.id,
        "doctor_id": note.doctor_id,
        "screening_session_id": note.screening_session_id,
        "note": note.note,
        "created_at": note.created_at.isoformat(),
    }


def _serialize_recommendation(item: Recommendation) -> dict[str, Any]:
    return {
        "id": item.id,
        "doctor_id": item.doctor_id,
        "screening_session_id": item.screening_session_id,
        "recommendation": item.recommendation,
        "created_at": item.created_at.isoformat(),
    }


@app.get("/api/patient/doctors")
def available_doctors_for_patient(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    profile = _patient_profile_for_user(db, current_user)
    rows = (
        db.query(User, DoctorProfile)
        .join(DoctorProfile, DoctorProfile.user_id == User.id)
        .filter(
            User.account_status == "active",
            DoctorProfile.verification_status == "verified",
        )
        .order_by(User.full_name.asc())
        .all()
    )
    existing = {
        relationship.doctor_id: relationship.consent_status
        for relationship in db.query(DoctorPatient).filter(
            DoctorPatient.patient_id == profile.id
        ).all()
    }
    return {
        "doctors": [
            {
                "user_id": user.id,
                "full_name": user.full_name,
                "specialization": doctor.specialization,
                "hospital": doctor.hospital,
                "verification_status": doctor.verification_status,
                "consent_status": existing.get(doctor.id),
            }
            for user, doctor in rows
        ]
    }


@app.post("/api/patient/doctors/{doctor_user_id}/consent")
def share_patient_record_with_doctor(
    doctor_user_id: int,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient_profile = _patient_profile_for_user(db, current_user)
    doctor_user = db.query(User).filter(
        User.id == doctor_user_id,
        User.role == "doctor",
        User.account_status == "active",
    ).first()
    doctor_profile = db.query(DoctorProfile).filter(
        DoctorProfile.user_id == doctor_user_id,
        DoctorProfile.verification_status == "verified",
    ).first()
    if doctor_user is None or doctor_profile is None:
        raise HTTPException(status_code=404, detail="Verified doctor not found.")

    relationship = db.query(DoctorPatient).filter(
        DoctorPatient.doctor_id == doctor_profile.id,
        DoctorPatient.patient_id == patient_profile.id,
    ).first()
    now = datetime.utcnow()
    if relationship is None:
        relationship = DoctorPatient(
            doctor_id=doctor_profile.id,
            patient_id=patient_profile.id,
            consent_status="consented",
            relationship_status="active",
            requested_at=now,
            consented_at=now,
            revoked_at=None,
        )
        db.add(relationship)
    else:
        relationship.consent_status = "consented"
        relationship.relationship_status = "active"
        relationship.consented_at = now
        relationship.revoked_at = None

    db.flush()
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action="patient_consent_granted",
        resource_type="doctor_patient",
        resource_id=str(relationship.id),
    ))
    db.commit()
    return {"message": "Patient record sharing is enabled for this doctor."}


@app.delete("/api/patient/doctors/{doctor_user_id}/consent")
def revoke_patient_record_consent(
    doctor_user_id: int,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    patient_profile = _patient_profile_for_user(db, current_user)
    doctor_profile = db.query(DoctorProfile).filter(
        DoctorProfile.user_id == doctor_user_id
    ).first()
    if doctor_profile is None:
        raise HTTPException(status_code=404, detail="Doctor not found.")
    relationship = db.query(DoctorPatient).filter(
        DoctorPatient.doctor_id == doctor_profile.id,
        DoctorPatient.patient_id == patient_profile.id,
    ).first()
    if relationship is None:
        raise HTTPException(status_code=404, detail="Doctor relationship not found.")

    relationship.consent_status = "revoked"
    relationship.relationship_status = "inactive"
    relationship.revoked_at = datetime.utcnow()
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action="patient_consent_revoked",
        resource_type="doctor_patient",
        resource_id=str(relationship.id),
    ))
    db.commit()
    return {"message": "Patient record sharing has been revoked."}


@app.get("/api/doctor/dashboard")
def doctor_dashboard(
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db),
):
    doctor = _doctor_profile_for_user(db, current_user)
    patient_ids = select(DoctorPatient.patient_id).where(
        DoctorPatient.doctor_id == doctor.id,
        DoctorPatient.consent_status == "consented",
        DoctorPatient.relationship_status == "active",
    )
    total_patients = db.query(DoctorPatient).filter(
        DoctorPatient.doctor_id == doctor.id,
        DoctorPatient.consent_status == "consented",
        DoctorPatient.relationship_status == "active",
    ).count()
    total_assessments = db.query(ScreeningSession).filter(
        ScreeningSession.patient_id.in_(patient_ids)
    ).count()
    review_count = db.query(ScreeningResult).join(
        ScreeningSession,
        ScreeningSession.id == ScreeningResult.screening_session_id,
    ).filter(
        ScreeningSession.patient_id.in_(patient_ids),
        ScreeningResult.predicted_class == "Depressed",
    ).count()
    recent = db.query(ScreeningSession).filter(
        ScreeningSession.patient_id.in_(patient_ids)
    ).order_by(ScreeningSession.created_at.desc()).limit(8).all()
    recent_payload = []
    for item in recent:
        latest_result = db.query(ScreeningResult).filter(
            ScreeningResult.screening_session_id == item.id
        ).order_by(ScreeningResult.created_at.desc()).first()
        recent_payload.append({
            "id": item.id,
            "patient_id": item.patient_id,
            "patient_name": (
                item.patient.user.full_name
                if item.patient and item.patient.user
                else "Patient"
            ),
            "screening_date": item.screening_date.isoformat(),
            "screening_status": item.screening_status,
            "completion_percentage": item.completion_percentage,
            "predicted_class": latest_result.predicted_class if latest_result else None,
        })

    return {
        "user": serialize_user(current_user),
        "metrics": {
            "assigned_patients": total_patients,
            "submitted_assessments": total_assessments,
            "depressed_classifications_for_review": review_count,
        },
        "recent_assessments": recent_payload,
    }


@app.get("/api/doctor/patients")
def list_doctor_patients(
    search: str = "",
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db),
):
    doctor = _doctor_profile_for_user(db, current_user)
    query = (
        db.query(PatientProfile, User)
        .join(User, User.id == PatientProfile.user_id)
        .join(DoctorPatient, DoctorPatient.patient_id == PatientProfile.id)
        .filter(
            DoctorPatient.doctor_id == doctor.id,
            DoctorPatient.consent_status == "consented",
            DoctorPatient.relationship_status == "active",
            User.role == "patient",
        )
    )
    if search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(
            User.full_name.ilike(term) | User.email.ilike(term)
        )

    output = []
    for profile, user in query.order_by(User.full_name.asc()).all():
        latest = db.query(ScreeningSession).filter(
            ScreeningSession.patient_id == profile.id
        ).order_by(ScreeningSession.created_at.desc()).first()
        latest_result = (
            db.query(ScreeningResult)
            .join(
                ScreeningSession,
                ScreeningSession.id == ScreeningResult.screening_session_id,
            )
            .filter(ScreeningSession.id == latest.id)
            .order_by(ScreeningResult.created_at.desc())
            .first() if latest else None
        )
        output.append({
            "patient_id": profile.id,
            "full_name": user.full_name,
            "age": profile.age,
            "gestational_age": profile.gestational_age,
            "latest_assessment_date": latest.screening_date.isoformat() if latest else None,
            "latest_assessment_status": latest.screening_status if latest else "not_started",
            "latest_prediction": latest_result.predicted_class if latest_result else None,
        })
    return {"patients": output}


@app.get("/api/doctor/patients/{patient_id}/history")
def doctor_patient_history(
    patient_id: int,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db),
):
    doctor = _doctor_profile_for_user(db, current_user)
    _consented_relationship(db, doctor, patient_id)
    profile = db.query(PatientProfile).filter(
        PatientProfile.id == patient_id
    ).first()
    if profile is None or profile.user is None:
        raise HTTPException(status_code=404, detail="Patient record not found.")
    sessions = db.query(ScreeningSession).filter(
        ScreeningSession.patient_id == profile.id
    ).order_by(ScreeningSession.created_at.desc()).all()
    notes = db.query(ClinicalNote).filter(
        ClinicalNote.doctor_id == doctor.id,
        ClinicalNote.patient_id == profile.id,
    ).order_by(ClinicalNote.created_at.desc()).all()
    recommendations = db.query(Recommendation).filter(
        Recommendation.doctor_id == doctor.id,
        Recommendation.patient_id == profile.id,
    ).order_by(Recommendation.created_at.desc()).all()
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action="doctor_viewed_patient_history",
        resource_type="patient_profile",
        resource_id=str(profile.id),
    ))
    db.commit()
    return {
        "patient": {
            "id": profile.id,
            "full_name": profile.user.full_name,
            "email": profile.user.email,
            "age": profile.age,
            "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
            "gender": profile.gender,
            "contact_phone": profile.contact_phone,
            "address": profile.address,
            "gestational_age": profile.gestational_age,
            "city": profile.city,
            "medical_history": profile.notes,
        },
        "screenings": [serialize_screening_session(item, db) for item in sessions],
        "clinical_notes": [_serialize_clinical_note(item) for item in notes],
        "recommendations": [_serialize_recommendation(item) for item in recommendations],
    }


def _validate_doctor_assessment_link(
    db: Session,
    patient_id: int,
    screening_session_id: int | None,
) -> None:
    if screening_session_id is None:
        return
    session = db.query(ScreeningSession).filter(
        ScreeningSession.id == screening_session_id,
        ScreeningSession.patient_id == patient_id,
    ).first()
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Assessment not found for this patient.",
        )


@app.post("/api/doctor/patients/{patient_id}/notes")
def add_clinical_note(
    patient_id: int,
    payload: ClinicalNoteRequest,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db),
):
    doctor = _doctor_profile_for_user(db, current_user)
    _consented_relationship(db, doctor, patient_id)
    note_text = payload.note.strip()
    if not note_text:
        raise HTTPException(status_code=422, detail="Clinical note cannot be empty.")
    _validate_doctor_assessment_link(db, patient_id, payload.screening_session_id)
    note = ClinicalNote(
        doctor_id=doctor.id,
        patient_id=patient_id,
        screening_session_id=payload.screening_session_id,
        note=note_text,
    )
    db.add(note)
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action="clinical_note_added",
        resource_type="patient_profile",
        resource_id=str(patient_id),
    ))
    db.commit()
    db.refresh(note)
    return _serialize_clinical_note(note)


@app.post("/api/doctor/patients/{patient_id}/recommendations")
def add_patient_recommendation(
    patient_id: int,
    payload: RecommendationRequest,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db),
):
    doctor = _doctor_profile_for_user(db, current_user)
    _consented_relationship(db, doctor, patient_id)
    recommendation_text = payload.recommendation.strip()
    if not recommendation_text:
        raise HTTPException(status_code=422, detail="Recommendation cannot be empty.")
    _validate_doctor_assessment_link(db, patient_id, payload.screening_session_id)
    recommendation = Recommendation(
        doctor_id=doctor.id,
        patient_id=patient_id,
        screening_session_id=payload.screening_session_id,
        recommendation=recommendation_text,
    )
    db.add(recommendation)
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action="patient_recommendation_added",
        resource_type="patient_profile",
        resource_id=str(patient_id),
    ))
    db.commit()
    db.refresh(recommendation)
    return _serialize_recommendation(recommendation)


@app.get("/api/patient/dashboard")
def patient_dashboard(current_user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    sessions = db.query(ScreeningSession).filter(ScreeningSession.patient_id == profile.id).order_by(ScreeningSession.created_at.desc()).all() if profile else []
    screening_data = [serialize_screening_session(session, db) for session in sessions]
    notes = db.query(ClinicalNote).filter(
        ClinicalNote.patient_id == profile.id
    ).order_by(ClinicalNote.created_at.desc()).all() if profile else []
    recommendations = db.query(Recommendation).filter(
        Recommendation.patient_id == profile.id
    ).order_by(Recommendation.created_at.desc()).all() if profile else []
    return {
        "user": serialize_user(current_user),
        "profile": {
            "id": profile.id if profile else None,
            "patient_reference": profile.patient_reference if profile else None,
            "age": profile.age if profile else None,
            "gestational_age": profile.gestational_age if profile else None,
            "city": profile.city if profile else None,
            "contact_phone": profile.contact_phone if profile else None,
            "date_of_birth": profile.date_of_birth.isoformat() if profile and profile.date_of_birth else None,
            "gender": profile.gender if profile else None,
            "address": profile.address if profile else None,
            "notes": profile.notes if profile else None,
            "user": serialize_user(current_user),
        } if profile else {"user": serialize_user(current_user)},
        "screenings": screening_data,
        "clinical_notes": [_serialize_clinical_note(item) for item in notes],
        "recommendations": [_serialize_recommendation(item) for item in recommendations],
    }


@app.put("/api/patient/profile")
def update_patient_profile(
    payload: PatientProfileUpdate,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
):
    profile = _patient_profile_for_user(db, current_user)
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="Provide at least one profile field to update.")
    for field, value in changes.items():
        if isinstance(value, str):
            value = value.strip() or None
        setattr(profile, field, value)
    db.add(AuditLog(
        actor_user_id=current_user.id,
        action="patient_profile_updated",
        resource_type="patient_profile",
        resource_id=str(profile.id),
    ))
    db.commit()
    return {
        "message": "Patient profile updated.",
        "profile": {
            "id": profile.id,
            "patient_reference": profile.patient_reference,
            "age": profile.age,
            "gestational_age": profile.gestational_age,
            "contact_phone": profile.contact_phone,
            "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
            "gender": profile.gender,
            "address": profile.address,
            "city": profile.city,
            "notes": profile.notes,
        },
    }


@app.get("/api/patient/history")
def patient_history(current_user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not profile:
        return {"user": serialize_user(current_user), "screenings": []}
    sessions = db.query(ScreeningSession).filter(ScreeningSession.patient_id == profile.id).order_by(ScreeningSession.created_at.desc()).all()
    return {
        "user": serialize_user(current_user),
        "screenings": [serialize_screening_session(session, db) for session in sessions],
    }


@app.post("/api/patient/screening/save")
def save_patient_screening(payload: ScreeningSaveRequest, current_user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    return {
        "message": "Screening submitted and saved.",
        **predict_from_form(
            ScreeningFormSubmission(answers=payload.answers),
            current_user,
            db,
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
