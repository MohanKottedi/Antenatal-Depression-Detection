import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
from xml.sax.saxutils import escape
from _thread import LockType

import joblib
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic import ValidationError

from model import FTTransformer, NUMERIC_FEATURE_NAMES, RAW_FEATURE_NAMES
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
app.mount(
    "/static",
    StaticFiles(directory=frontend_dir),
    name="static",
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


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(frontend_dir / "index.html")


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
    azure_request = Request(
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

    return Response(
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

class FeatureVector(BaseModel):
    age: int = 25
    gestational_age_weeks: int = 20
    first_pregnancy: bool = True
    number_of_children: int = 0
    sleep_quality: str = "average"
    stress_level: str = "medium"
    sadness_frequency: str = "sometimes"

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
        "risk_score": risk_proba
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
