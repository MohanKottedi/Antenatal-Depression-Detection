from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import TARGET_MAPPING

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svm_experiment_A.joblib"

if MODEL_PATH.exists():
    MODEL_PACKAGE = joblib.load(MODEL_PATH)
    model = MODEL_PACKAGE.get("model")
    FEATURE_NAMES = list(MODEL_PACKAGE.get("feature_names", []))
else:
    MODEL_PACKAGE = None
    model = None
    FEATURE_NAMES = []


def get_model_info() -> dict:
    return {
        "model": MODEL_PACKAGE.get("model_name", "SVM") if MODEL_PACKAGE else "SVM",
        "experiment": MODEL_PACKAGE.get("experiment", "Experiment A") if MODEL_PACKAGE else "Experiment A",
        "feature_names": FEATURE_NAMES,
    }


def predict_antenatal_depression(answers: dict) -> dict:
    if model is None:
        raise FileNotFoundError(f"SVM model not found: {MODEL_PATH}")

    if not isinstance(answers, dict):
        raise TypeError("answers must be a dictionary.")

    missing = [name for name in FEATURE_NAMES if name not in answers or answers[name] in (None, "")]
    if missing:
        raise ValueError(f"Missing required questionnaire answers: {missing}")

    row = {name: answers[name] for name in FEATURE_NAMES}
    frame = pd.DataFrame([row], columns=FEATURE_NAMES)
    transformed = frame.to_numpy(dtype=float)
    if not np.isfinite(transformed).all():
        raise ValueError("Numeric questionnaire answers must be finite.")

    prediction = int(model.predict(transformed)[0])
    probabilities = model.predict_proba(transformed)[0]
    class_labels = {value: name for name, value in TARGET_MAPPING.items()}
    class_probabilities = {
        "Not Depressed": round(float(probabilities[0]), 6) if 0 in class_labels else 0.0,
        "Depressed": round(float(probabilities[1]), 6) if 1 in class_labels else 0.0,
    }

    if prediction == 1:
        probability = float(probabilities[list(model.classes_).index(1)]) if 1 in model.classes_ else 0.0
    else:
        probability = float(probabilities[list(model.classes_).index(0)]) if 0 in model.classes_ else 0.0

    return {
        "model": "SVM",
        "experiment": "Experiment A",
        "prediction": prediction,
        "label": "Depressed" if prediction == 1 else "Not Depressed",
        "probability": round(probability, 6),
        "class_probabilities": class_probabilities,
    }
