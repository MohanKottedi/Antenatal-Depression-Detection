from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

SPLIT_PATH = MODEL_DIR / "data_splits.joblib"

OUTPUT_PATH = (
    MODEL_DIR / "transformer_data_experiment_A.joblib"
)

METADATA_PATH = (
    MODEL_DIR / "transformer_metadata_experiment_A.json"
)


# ============================================================
# FEATURES
# ============================================================
from config import (
    RISK_FACTOR_COLUMNS,
    SYMPTOM_COLUMNS,
)

NUMERIC_COLUMNS = [
    "Age",
    "Gestational Age",
    "Number of sons",
    "Number of daughters",
    "Total Number of Children",
]

CATEGORICAL_COLUMNS = [
    column
    for column in RISK_FACTOR_COLUMNS
    if column not in NUMERIC_COLUMNS
]

ALL_FEATURES = (
    RISK_FACTOR_COLUMNS
    + SYMPTOM_COLUMNS
)


# ============================================================
# HELPERS
# ============================================================

def clean_string_value(value):
    if pd.isna(value):
        return "__MISSING__"

    value = str(value).strip()

    if value == "":
        return "__MISSING__"

    return value


def build_category_mapping(series):
    values = sorted(
        {
            clean_string_value(value)
            for value in series
        }
    )

    return {
        value: index + 1
        for index, value in enumerate(values)
    }


def encode_categorical(series, mapping):
    encoded = []

    for value in series:
        value = clean_string_value(value)
        encoded.append(
            mapping.get(value, 0)
        )

    return np.asarray(
        encoded,
        dtype=np.int64,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING TRANSFORMER DATA - EXPERIMENT A")
    print("25 Features / Native Categorical Embeddings")
    print("=" * 70)

    # --------------------------------------------------------
    # Load existing split
    # --------------------------------------------------------

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_PATH}"
        )

    split_data = joblib.load(
        SPLIT_PATH
    )

    print("\nLoaded split:")
    print(SPLIT_PATH)

    # --------------------------------------------------------
    # Exact 25-feature column order
    # --------------------------------------------------------

    feature_columns = ALL_FEATURES

    X_train = pd.DataFrame(
        split_data["X_train"],
        columns=feature_columns,
    )
    # Verify every Experiment A feature exists correctly.
    missing_features = [
        column
        for column in ALL_FEATURES
        if column not in X_train.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing Experiment A features:\n"
            + "\n".join(missing_features)
        )

    # Verify symptom columns actually contain data.
    # for column in SYMPTOM_COLUMNS:
    #     observed_count = X_train[column].notna().sum()

    #     if observed_count == 0:
    #         raise ValueError(
    #             f"Symptom column has no observed values: {column!r}\n"
    #             "Check the column name against the original dataset."
    #         )

    X_val = pd.DataFrame(
        split_data["X_val"],
        columns=feature_columns,
    )

    X_test = pd.DataFrame(
        split_data["X_test"],
        columns=feature_columns,
    )

    y_train = np.asarray(
        split_data["y_train"],
        dtype=np.int64,
    )

    y_val = np.asarray(
        split_data["y_val"],
        dtype=np.int64,
    )

    y_test = np.asarray(
        split_data["y_test"],
        dtype=np.int64,
    )

    print("\nExperiment A:")
    print(f"Numerical features   : {len(NUMERIC_COLUMNS)}")
    print(f"Categorical features : {len(CATEGORICAL_COLUMNS)}")
    print(f"Symptom features     : {len(SYMPTOM_COLUMNS)}")
    print(f"Total features       : {len(ALL_FEATURES)}")

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    print("\nPreparing numerical features...")

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_numeric = numeric_imputer.fit_transform(
        X_train[NUMERIC_COLUMNS]
    )

    X_val_numeric = numeric_imputer.transform(
        X_val[NUMERIC_COLUMNS]
    )

    X_test_numeric = numeric_imputer.transform(
        X_test[NUMERIC_COLUMNS]
    )

    scaler = StandardScaler()

    X_train_numeric = scaler.fit_transform(
        X_train_numeric
    )

    X_val_numeric = scaler.transform(
        X_val_numeric
    )

    X_test_numeric = scaler.transform(
        X_test_numeric
    )

    X_train_numeric = np.asarray(
        X_train_numeric,
        dtype=np.float32,
    )

    X_val_numeric = np.asarray(
        X_val_numeric,
        dtype=np.float32,
    )

    X_test_numeric = np.asarray(
        X_test_numeric,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    print("\nPreparing categorical features...")

    category_mappings = {}
    category_cardinalities = {}

    train_categorical_arrays = []
    val_categorical_arrays = []
    test_categorical_arrays = []

    for column in CATEGORICAL_COLUMNS:

        mapping = build_category_mapping(
            X_train[column]
        )

        category_mappings[column] = mapping

        # ID 0 is reserved for unknown/missing.
        cardinality = len(mapping) + 1

        category_cardinalities[column] = (
            cardinality
        )

        train_encoded = encode_categorical(
            X_train[column],
            mapping,
        )

        val_encoded = encode_categorical(
            X_val[column],
            mapping,
        )

        test_encoded = encode_categorical(
            X_test[column],
            mapping,
        )

        train_categorical_arrays.append(
            train_encoded
        )

        val_categorical_arrays.append(
            val_encoded
        )

        test_categorical_arrays.append(
            test_encoded
        )

        print(
            f"{column}: "
            f"{len(mapping)} categories "
            f"(vocabulary={cardinality})"
        )

    X_train_categorical = np.column_stack(
        train_categorical_arrays
    ).astype(np.int64)

    X_val_categorical = np.column_stack(
        val_categorical_arrays
    ).astype(np.int64)

    X_test_categorical = np.column_stack(
        test_categorical_arrays
    ).astype(np.int64)

    # --------------------------------------------------------
    # Symptom preprocessing
    # --------------------------------------------------------
    #
    # Symptoms are numerical ordinal scores.
    # We treat each symptom as an individual numerical token.
    #
    # We DO NOT combine them into Scalling.
    # Scalling is excluded to avoid leakage.
    # --------------------------------------------------------

    print("\nPreparing symptom features...")

    symptom_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_symptoms = symptom_imputer.fit_transform(
        X_train[SYMPTOM_COLUMNS]
    )

    X_val_symptoms = symptom_imputer.transform(
        X_val[SYMPTOM_COLUMNS]
    )

    X_test_symptoms = symptom_imputer.transform(
        X_test[SYMPTOM_COLUMNS]
    )

    symptom_scaler = StandardScaler()

    X_train_symptoms = symptom_scaler.fit_transform(
        X_train_symptoms
    )

    X_val_symptoms = symptom_scaler.transform(
        X_val_symptoms
    )

    X_test_symptoms = symptom_scaler.transform(
        X_test_symptoms
    )

    X_train_symptoms = np.asarray(
        X_train_symptoms,
        dtype=np.float32,
    )

    X_val_symptoms = np.asarray(
        X_val_symptoms,
        dtype=np.float32,
    )

    X_test_symptoms = np.asarray(
        X_test_symptoms,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidating Transformer data...")

    assert X_train_numeric.shape == (
        len(y_train),
        len(NUMERIC_COLUMNS),
    )

    assert X_val_numeric.shape == (
        len(y_val),
        len(NUMERIC_COLUMNS),
    )

    assert X_test_numeric.shape == (
        len(y_test),
        len(NUMERIC_COLUMNS),
    )

    assert X_train_symptoms.shape == (
        len(y_train),
        len(SYMPTOM_COLUMNS),
    )

    assert X_val_symptoms.shape == (
        len(y_val),
        len(SYMPTOM_COLUMNS),
    )

    assert X_test_symptoms.shape == (
        len(y_test),
        len(SYMPTOM_COLUMNS),
    )

    assert X_train_categorical.shape == (
        len(y_train),
        len(CATEGORICAL_COLUMNS),
    )

    assert X_val_categorical.shape == (
        len(y_val),
        len(CATEGORICAL_COLUMNS),
    )

    assert X_test_categorical.shape == (
        len(y_test),
        len(CATEGORICAL_COLUMNS),
    )

    assert np.isfinite(
        X_train_numeric
    ).all()

    assert np.isfinite(
        X_train_symptoms
    ).all()

    assert np.isfinite(
        X_val_numeric
    ).all()

    assert np.isfinite(
        X_val_symptoms
    ).all()

    assert np.isfinite(
        X_test_numeric
    ).all()

    assert np.isfinite(
        X_test_symptoms
    ).all()

    assert np.all(
        X_train_categorical >= 0
    )

    assert np.all(
        X_val_categorical >= 0
    )

    assert np.all(
        X_test_categorical >= 0
    )

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------

    transformer_data = {

        "X_train_numeric":
            X_train_numeric,

        "X_val_numeric":
            X_val_numeric,

        "X_test_numeric":
            X_test_numeric,

        "X_train_categorical":
            X_train_categorical,

        "X_val_categorical":
            X_val_categorical,

        "X_test_categorical":
            X_test_categorical,

        "X_train_symptoms":
            X_train_symptoms,

        "X_val_symptoms":
            X_val_symptoms,

        "X_test_symptoms":
            X_test_symptoms,

        "y_train":
            y_train,

        "y_val":
            y_val,

        "y_test":
            y_test,

        "numeric_columns":
            NUMERIC_COLUMNS,

        "categorical_columns":
            CATEGORICAL_COLUMNS,

        "symptom_columns":
            SYMPTOM_COLUMNS,

        "all_features":
            ALL_FEATURES,

        "numeric_imputer":
            numeric_imputer,

        "numeric_scaler":
            scaler,

        "symptom_imputer":
            symptom_imputer,

        "symptom_scaler":
            symptom_scaler,

        "category_mappings":
            category_mappings,

        "category_cardinalities":
            category_cardinalities,
    }

    joblib.dump(
        transformer_data,
        OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "experiment": "A",

        "description": (
            "25-feature native categorical "
            "tabular Transformer dataset"
        ),

        "numeric_features":
            NUMERIC_COLUMNS,

        "categorical_features":
            CATEGORICAL_COLUMNS,

        "symptom_features":
            SYMPTOM_COLUMNS,

        "num_numeric_features":
            len(NUMERIC_COLUMNS),

        "num_categorical_features":
            len(CATEGORICAL_COLUMNS),

        "num_symptom_features":
            len(SYMPTOM_COLUMNS),

        "num_total_features":
            len(ALL_FEATURES),

        "train_numeric_shape":
            list(X_train_numeric.shape),

        "validation_numeric_shape":
            list(X_val_numeric.shape),

        "test_numeric_shape":
            list(X_test_numeric.shape),

        "train_categorical_shape":
            list(X_train_categorical.shape),

        "validation_categorical_shape":
            list(X_val_categorical.shape),

        "test_categorical_shape":
            list(X_test_categorical.shape),

        "train_symptom_shape":
            list(X_train_symptoms.shape),

        "validation_symptom_shape":
            list(X_val_symptoms.shape),

        "test_symptom_shape":
            list(X_test_symptoms.shape),

        "category_cardinalities":
            category_cardinalities,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRANSFORMER EXPERIMENT A DATA READY")
    print("=" * 70)

    print("\nNumerical tensors:")
    print(
        f"Train: {X_train_numeric.shape}"
    )
    print(
        f"Val  : {X_val_numeric.shape}"
    )
    print(
        f"Test : {X_test_numeric.shape}"
    )

    print("\nCategorical tensors:")
    print(
        f"Train: {X_train_categorical.shape}"
    )
    print(
        f"Val  : {X_val_categorical.shape}"
    )
    print(
        f"Test : {X_test_categorical.shape}"
    )

    print("\nSymptom tensors:")
    print(
        f"Train: {X_train_symptoms.shape}"
    )
    print(
        f"Val  : {X_val_symptoms.shape}"
    )
    print(
        f"Test : {X_test_symptoms.shape}"
    )

    print("\nLabels:")
    print(
        f"Train: {y_train.shape}"
    )
    print(
        f"Val  : {y_val.shape}"
    )
    print(
        f"Test : {y_test.shape}"
    )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()