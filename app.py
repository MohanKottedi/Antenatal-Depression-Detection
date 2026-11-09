import os
import re
from dataclasses import dataclass
from datetime import datetime
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
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse, Response as FastAPIResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, EmailStr, Field, ValidationError
from sqlalchemy.orm import Session

from model import FTTransformer, NUMERIC_FEATURE_NAMES, RAW_FEATURE_NAMES
from src.auth_service import COOKIE_NAME, create_access_token, decode_access_token, get_current_user, hash_password, require_role, verify_password
from src.database import AuditLog, PatientProfile, ScreeningResponse, ScreeningResult, ScreeningSession, SessionLocal, User, create_audit_entry, ensure_admin_account, get_db, init_db
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
base_dir = Path(__file__).resolve().parent
frontend_dir = base_dir / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

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
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str
    consent: bool = False

    model_config = ConfigDict(str_strip_whitespace=True)

    def validate_password_match(self) -> None:
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        if len(self.password) < 8:
            raise ValueError("Password must be at least 8 characters long.")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


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
    init_db()
    db = SessionLocal()
    try:
        ensure_admin_account(db)
    finally:
        db.close()


def frontend() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/", include_in_schema=False, response_model=None)
def frontend_root(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        try:
            payload = decode_access_token(token)
            user_id = payload.get("user_id")
            if user_id is not None:
                db = SessionLocal()
                try:
                    user = db.query(User).filter(User.id == user_id).first()
                    if user and user.role == "patient":
                        return RedirectResponse(url="/patient", status_code=302)
                finally:
                    db.close()
        except HTTPException:
            pass
    return frontend()


@app.get("/auth/login", include_in_schema=False)
def auth_login() -> FileResponse:
    return FileResponse(frontend_dir / "login.html")


@app.get("/auth/register", include_in_schema=False)
def auth_register() -> FileResponse:
    return FileResponse(frontend_dir / "register.html")


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(frontend_dir / "admin.html")


@app.get("/patient", include_in_schema=False)
def patient_page() -> FileResponse:
    return FileResponse(frontend_dir / "patient.html")


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


@app.post("/api/form/predict")
def predict_from_form(request: ScreeningFormSubmission):
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

    return {"screening_result": _to_json_compatible(screening_result)}


@app.post("/api/conversation/start")
def start_conversation():
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
def respond_to_conversation(request: ConversationMessage):
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


def build_depression_summary(predicted_class: str | None, risk_score: float | None) -> str:
    score = float(risk_score) if risk_score is not None else 0.0
    if predicted_class == "Depressed":
        if score >= 0.7:
            return (
                "High risk: the screening indicates a likely depressive risk. This result should be reviewed by a clinician and discussed with the patient as a follow-up opportunity."
            )
        if score >= 0.4:
            return (
                "Moderate risk: the screening suggests elevated depressive symptoms and warrants supportive follow-up, symptom monitoring, and professional evaluation."
            )
        return (
            "Low-to-moderate risk: depressive symptoms are present but not strongly suggestive of a high-risk profile. Supportive follow-up and routine monitoring are recommended."
        )
    if predicted_class == "Not Depressed":
        return (
            "Low risk: the screening result does not currently indicate a depressive-risk pattern based on the provided answers. Continue regular monitoring and routine follow-up."
        )
    return (
        "Assessment in progress: no final depression prediction was available at the time of saving. Review the patient answers and complete the screening for a clinically informed result."
    )


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

    user = User(
        full_name=payload.full_name.strip(),
        email=normalized_email,
        password_hash=hash_password(payload.password),
        role="patient",
        account_status="active",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    profile = PatientProfile(
        user_id=user.id,
        patient_reference=f"SN-{uuid4().hex[:8].upper()}",
        age=None,
        gestational_age=None,
    )
    db.add(profile)
    db.commit()
    create_audit_entry(db, "user_registered", "user", str(user.id), actor_user_id=user.id)
    return {"message": "Registration successful.", "user": serialize_user(user)}


@app.post("/api/login")
def login_user(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    normalized_email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == normalized_email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if user.account_status != "active":
        raise HTTPException(status_code=403, detail="This account is currently inactive.")

    token = create_access_token(user.email, user.id, user.role)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 12,
    )
    user.last_login_at = datetime.utcnow()
    db.commit()
    create_audit_entry(db, "user_login", "user", str(user.id), actor_user_id=user.id)
    return {"message": "Login successful.", "user": serialize_user(user)}


@app.post("/api/logout")
def logout_user(response: Response):
    response.delete_cookie(key=COOKIE_NAME, httponly=True, samesite="lax")
    return {"message": "Logout successful."}


@app.get("/api/me")
def get_current_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    return {
        "user": serialize_user(current_user),
        "profile": {
            "id": profile.id if profile else None,
            "patient_reference": profile.patient_reference if profile else None,
            "age": profile.age if profile else None,
            "gestational_age": profile.gestational_age if profile else None,
            "city": profile.city if profile else None,
            "contact_phone": profile.contact_phone if profile else None,
        } if profile else None,
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
    db.query(ScreeningResponse).delete()
    db.query(ScreeningResult).delete()
    db.query(ScreeningSession).delete()
    db.query(PatientProfile).delete()
    db.commit()
    create_audit_entry(db, "screening_data_cleared", "system", "all", actor_user_id=current_user.id)
    return {"message": "Saved screening data and patient analytics have been cleared."}


@app.get("/api/patient/dashboard")
def patient_dashboard(current_user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    sessions = db.query(ScreeningSession).filter(ScreeningSession.patient_id == profile.id).order_by(ScreeningSession.created_at.desc()).all() if profile else []
    screening_data = [serialize_screening_session(session, db) for session in sessions]
    return {
        "user": serialize_user(current_user),
        "profile": {
            "id": profile.id if profile else None,
            "patient_reference": profile.patient_reference if profile else None,
            "age": profile.age if profile else None,
            "gestational_age": profile.gestational_age if profile else None,
            "city": profile.city if profile else None,
            "contact_phone": profile.contact_phone if profile else None,
            "user": serialize_user(current_user),
        } if profile else {"user": serialize_user(current_user)},
        "screenings": screening_data,
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
    profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not profile:
        profile = PatientProfile(user_id=current_user.id, patient_reference=f"SN-{uuid4().hex[:8].upper()}")
        db.add(profile)
        db.commit()
        db.refresh(profile)

    summary = payload.screening_summary or build_depression_summary(payload.predicted_class, payload.risk_score)

    session = ScreeningSession(
        patient_id=profile.id,
        screening_date=datetime.utcnow(),
        screening_status=payload.screening_status,
        completion_percentage=float(payload.completion_percentage),
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    for key, value in payload.answers.items():
        response = ScreeningResponse(screening_session_id=session.id, question_key=str(key), response_value=str(value))
        db.add(response)

    if payload.predicted_class or payload.risk_score is not None:
        result = ScreeningResult(
            screening_session_id=session.id,
            model_name=payload.model_name,
            model_version=payload.model_version,
            predicted_class=payload.predicted_class or "Pending",
            risk_score=payload.risk_score,
            screening_summary=summary,
        )
        db.add(result)

    db.commit()
    create_audit_entry(db, "screening_saved", "screening_session", str(session.id), actor_user_id=current_user.id)
    return {
        "message": "Screening saved successfully.",
        "session_id": session.id,
        "screening_summary": summary,
        "predicted_class": payload.predicted_class or "Pending",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
