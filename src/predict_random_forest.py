
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import TARGET_MAPPING
from src.sneha.feature_extractor import CATEGORY_VALUES


# ============================================================
# PATHS AND MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_experiment_A.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor_experiment_A.joblib"

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
MISSING_CATEGORY_FEATURES = {
    name
    for name, values in CATEGORY_VALUES.items()
    if "__MISSING__" in values
}

if not all(isinstance(name, str) for name in FEATURE_NAMES):
    raise ValueError(
        "All feature names must be strings."
    )

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Random Forest preprocessor not found: {PREPROCESSOR_PATH}"
    )

preprocessor = joblib.load(PREPROCESSOR_PATH)
raw_input_names: Any = getattr(preprocessor, "feature_names_in_", None)

if (
    not isinstance(raw_input_names, (list, tuple, np.ndarray))
    or len(raw_input_names) == 0
):
    raise ValueError(
        "The saved Random Forest preprocessor has no valid input feature names."
    )

INPUT_FEATURE_NAMES: list[str] = list(raw_input_names)
if not all(isinstance(name, str) for name in INPUT_FEATURE_NAMES):
    raise ValueError("All preprocessor input feature names must be strings.")

transformed_names = list(preprocessor.get_feature_names_out())
if transformed_names != FEATURE_NAMES:
    raise ValueError(
        "The saved preprocessor output features do not match the Random "
        "Forest model package."
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_antenatal_depression(answers: dict) -> dict:
    """
    Predict using the existing Experiment A Random Forest.

    The answers are the original questionnaire values. The saved
    Experiment A preprocessor applies the training transformations.
    The questionnaire's explicit ``__MISSING__`` category is passed to the
    saved categorical imputer as a missing value and is reported in the result.
    No retraining is performed.
    """

    if not isinstance(answers, dict):
        raise TypeError("answers must be a dictionary.")

    missing = [
        name for name in INPUT_FEATURE_NAMES
        if name not in answers or answers[name] is None
        or answers[name] == ""
    ]

    if missing:
        raise ValueError(
            f"Missing required questionnaire answers: {missing}"
        )

    row = {name: answers[name] for name in INPUT_FEATURE_NAMES}
    raw_features = pd.DataFrame([row], columns=INPUT_FEATURE_NAMES)

    numeric_columns = []
    categorical_columns = []
    for name, _, columns in preprocessor.transformers_:
        if name == "numeric":
            numeric_columns = list(columns)
        elif name == "categorical":
            categorical_columns = list(columns)

    if not numeric_columns or not categorical_columns:
        raise ValueError(
            "The saved Random Forest preprocessor has an unexpected structure."
        )

    imputed_features = []
    for column in categorical_columns:
        if (
            column in MISSING_CATEGORY_FEATURES
            and raw_features.at[0, column] == "__MISSING__"
        ):
            raw_features.at[0, column] = np.nan
            imputed_features.append(column)

    if any(
        isinstance(raw_features.at[0, column], (bool, np.bool_))
        for column in numeric_columns
    ):
        raise ValueError(
            "Boolean values are not valid numeric questionnaire answers."
        )

    try:
        numeric_features = raw_features[numeric_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Random Forest numeric questionnaire answers must be numbers."
        ) from exc

    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("Numeric questionnaire answers must be finite.")

    categorical_pipeline = next(
        transformer
        for name, transformer, _ in preprocessor.transformers_
        if name == "categorical"
    )
    encoder = categorical_pipeline.named_steps["encoder"]
    for column, allowed_values in zip(
        categorical_columns,
        encoder.categories_,
    ):
        if pd.isna(raw_features.at[0, column]):
            continue
        if raw_features.at[0, column] not in allowed_values:
            raise ValueError(
                f"Unsupported category for {column}: "
                f"{raw_features.at[0, column]!r}"
            )

    transformed = preprocessor.transform(raw_features)
    if not np.isfinite(np.asarray(transformed, dtype=float)).all():
        raise ValueError("Preprocessed model inputs must be finite.")

    prediction = int(model.predict(transformed)[0])

    probability = None
    class_probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[0]
        classes = list(model.classes_)
        class_labels = {
            class_id: (
                "Not Depressed" if label == "Not" else label
            )
            for label, class_id in TARGET_MAPPING.items()
        }
        class_probabilities = {
            class_labels[int(class_id)]: round(float(probabilities[index]), 6)
            for index, class_id in enumerate(classes)
            if int(class_id) in class_labels
        }
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
        "class_probabilities": class_probabilities,
        "imputed_features": imputed_features,
    }


def get_model_info() -> dict:
    return {
        "model": MODEL_PACKAGE.get("model_name", "Random Forest"),
        "experiment": MODEL_PACKAGE.get("experiment", "Experiment A"),
        "feature_names": list(FEATURE_NAMES),
    }
