
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS AND MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_experiment_A.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Random Forest model package not found: {MODEL_PATH}"
    )

MODEL_PACKAGE = joblib.load(MODEL_PATH)

if not isinstance(MODEL_PACKAGE, dict) or "model" not in MODEL_PACKAGE:
    raise ValueError(
        "Unexpected Random Forest package format. "
        "Expected a dictionary containing 'model'."
    )

model = MODEL_PACKAGE["model"]

from typing import Any

raw_feature_names: Any = MODEL_PACKAGE.get("feature_names")

if (
    not isinstance(raw_feature_names, (list, tuple))
    or len(raw_feature_names) == 0
):
    raise ValueError(
        "The saved model package has no valid feature_names. "
        "Check the preprocessing package before prediction."
    )

FEATURE_NAMES: list[str] = list(raw_feature_names)

if not all(isinstance(name, str) for name in FEATURE_NAMES):
    raise ValueError(
        "All feature names must be strings."
    )
    


# ============================================================
# PREDICTION
# ============================================================

def predict_antenatal_depression(answers: dict) -> dict:
    """
    Predict using the existing Experiment A Random Forest.

    The answers must already be encoded and ordered exactly as
    the model's training features. No retraining is performed.
    """

    if not isinstance(answers, dict):
        raise TypeError("answers must be a dictionary.")

    missing = [
        name for name in FEATURE_NAMES
        if name not in answers or answers[name] is None
        or answers[name] == ""
    ]

    if missing:
        raise ValueError(
            f"Missing required Random Forest features: {missing}"
        )

    # Preserve the training feature order.
    row = {name: answers[name] for name in FEATURE_NAMES}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)

    # Reject non-numeric or non-finite model inputs.
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Random Forest expects numeric, preprocessed features. "
            "Use the same preprocessing and category encodings as training."
        ) from exc

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Model inputs must contain only finite values.")

    prediction = int(model.predict(X)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    return {
        "model": "Random Forest",
        "experiment": "A",
        "prediction": prediction,
        "label": (
            "Depressed" if prediction == 1 else "Not Depressed"
        ),
        "probability": (
            round(probability, 6)
            if probability is not None else None
        ),
    }


def get_model_info() -> dict:
    return {
        "model": MODEL_PACKAGE.get("model_name", "Random Forest"),
        "experiment": MODEL_PACKAGE.get("experiment", "Experiment A"),
        "feature_names": list(FEATURE_NAMES),
    }
