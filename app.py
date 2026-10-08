from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
import torch
from pydantic import ValidationError

from model import FTTransformer, NUMERIC_FEATURE_NAMES, RAW_FEATURE_NAMES

app = FastAPI(title="Antenatal Depression Screening API")

base_dir = Path(__file__).resolve().parent
ft_model_path = base_dir / "models" / "ft_transformer_antenatal_model.pkl"
legacy_model_path = base_dir / "models" / "screening_pipeline.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model = None
ft_checkpoint = None
pipeline = None
model_kind = "unavailable"
model_load_error = None

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
