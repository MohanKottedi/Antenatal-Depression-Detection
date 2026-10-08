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
OUTPUT_PATH = MODEL_DIR / "transformer_data_experiment_B.joblib"
METADATA_PATH = MODEL_DIR / "transformer_metadata_experiment_B.json"


# ============================================================
# FEATURES
# ============================================================

NUMERIC_COLUMNS = [
    "Age",
    "Gestational Age",
    "Number of sons",
    "Number of daughters",
    "Total Number of Children",
]

CATEGORICAL_COLUMNS = [
    "Gravida",
    "Female Education",
    "Husband Education",
    "Working Status",
    "Physical Health",
    "Previous Miscarriage",
    "Sufficient Money for Basic Needs",
    "Current Appereance Acceptance",
    "Family System",
    "Male Gender Preference",
    "Relationship with Mother in-law",
]

ALL_FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


# ============================================================
# HELPERS
# ============================================================

def clean_string_value(value):
    """Convert categorical values to clean strings."""
    if pd.isna(value):
        return "__MISSING__"

    value = str(value).strip()

    if value == "":
        return "__MISSING__"

    return value


def build_category_mapping(series):
    """
    Build a stable integer mapping for one categorical feature.

    0 is reserved for unknown/missing values.
    Actual categories start at 1.
    """
    values = sorted(
        {
            clean_string_value(value)
            for value in series
        }
    )

    mapping = {
        value: index + 1
        for index, value in enumerate(values)
    }

    return mapping


def encode_categorical(series, mapping):
    """
    Convert categorical values to integer IDs.

    Unknown values are encoded as 0.
    """
    encoded = []

    for value in series:
        value = clean_string_value(value)
        encoded.append(mapping.get(value, 0))

    return np.asarray(encoded, dtype=np.int64)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING TRANSFORMER DATA - EXPERIMENT B")
    print("Native Categorical Embeddings")
    print("=" * 70)

    # --------------------------------------------------------
    # Load original split
    # --------------------------------------------------------

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_PATH}"
        )

    split_data = joblib.load(SPLIT_PATH)

    print("\nLoaded split file:")
    print(SPLIT_PATH)

    # The existing split contains Experiment A features.
    # Convert them back to DataFrames so we can select
    # Experiment B risk factors only.

    feature_columns = [
        "Age",
        "Gestational Age",
        "Number of sons",
        "Number of daughters",
        "Total Number of Children",
        "Gravida",
        "Female Education",
        "Husband Education",
        "Working Status",
        "Physical Health",
        "Previous Miscarriage",
        "Sufficient Money for Basic Needs",
        "Current Appereance Acceptance",
        "Family System",
        "Male Gender Preference",
        "Relationship with Mother in-law",
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying sleep or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling badabout yourself that you are failure or let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "Moving or speaking so slowly that other people could have Noticed.",
        "Thoughts that you would be better off dead, or of hurting yourself",
    ]

    X_train_full = pd.DataFrame(
        split_data["X_train"],
        columns=feature_columns,
    )

    X_val_full = pd.DataFrame(
        split_data["X_val"],
        columns=feature_columns,
    )

    X_test_full = pd.DataFrame(
        split_data["X_test"],
        columns=feature_columns,
    )

    y_train = np.asarray(split_data["y_train"], dtype=np.int64)
    y_val = np.asarray(split_data["y_val"], dtype=np.int64)
    y_test = np.asarray(split_data["y_test"], dtype=np.int64)

    # --------------------------------------------------------
    # Select Experiment B features
    # --------------------------------------------------------

    X_train = X_train_full[ALL_FEATURES].copy()
    X_val = X_val_full[ALL_FEATURES].copy()
    X_test = X_test_full[ALL_FEATURES].copy()

    print("\nExperiment B features:")
    print(f"Numerical features   : {len(NUMERIC_COLUMNS)}")
    print(f"Categorical features : {len(CATEGORICAL_COLUMNS)}")
    print(f"Total features       : {len(ALL_FEATURES)}")

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    print("\nPreparing numerical features...")

    numeric_imputer = SimpleImputer(strategy="median")

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

        # +1 because ID 0 is reserved for unknown/missing.
        cardinality = len(mapping) + 1

        category_cardinalities[column] = cardinality

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

        train_categorical_arrays.append(train_encoded)
        val_categorical_arrays.append(val_encoded)
        test_categorical_arrays.append(test_encoded)

        print(
            f"{column}: "
            f"{len(mapping)} categories "
            f"(embedding vocabulary size={cardinality})"
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
    # Validate
    # --------------------------------------------------------

    print("\nValidating Transformer data...")

    assert X_train_numeric.shape[0] == len(y_train)
    assert X_val_numeric.shape[0] == len(y_val)
    assert X_test_numeric.shape[0] == len(y_test)

    assert X_train_categorical.shape[0] == len(y_train)
    assert X_val_categorical.shape[0] == len(y_val)
    assert X_test_categorical.shape[0] == len(y_test)

    assert np.isfinite(X_train_numeric).all()
    assert np.isfinite(X_val_numeric).all()
    assert np.isfinite(X_test_numeric).all()

    for array in [
        X_train_categorical,
        X_val_categorical,
        X_test_categorical,
    ]:
        assert np.all(array >= 0)

    # --------------------------------------------------------
    # Save everything needed for Transformer training
    # --------------------------------------------------------

    transformer_data = {
        "X_train_numeric": X_train_numeric,
        "X_val_numeric": X_val_numeric,
        "X_test_numeric": X_test_numeric,

        "X_train_categorical": X_train_categorical,
        "X_val_categorical": X_val_categorical,
        "X_test_categorical": X_test_categorical,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "all_features": ALL_FEATURES,

        "numeric_imputer": numeric_imputer,
        "scaler": scaler,

        "category_mappings": category_mappings,
        "category_cardinalities": category_cardinalities,
    }

    joblib.dump(
        transformer_data,
        OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Metadata JSON
    # --------------------------------------------------------

    metadata = {
        "experiment": "B",
        "description": (
            "Risk-factor-only native categorical tabular "
            "Transformer dataset"
        ),
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "num_numeric_features": len(NUMERIC_COLUMNS),
        "num_categorical_features": len(CATEGORICAL_COLUMNS),
        "num_total_features": len(ALL_FEATURES),
        "train_shape_numeric": list(X_train_numeric.shape),
        "validation_shape_numeric": list(X_val_numeric.shape),
        "test_shape_numeric": list(X_test_numeric.shape),
        "train_shape_categorical": list(
            X_train_categorical.shape
        ),
        "validation_shape_categorical": list(
            X_val_categorical.shape
        ),
        "test_shape_categorical": list(
            X_test_categorical.shape
        ),
        "category_cardinalities": category_cardinalities,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRANSFORMER DATA PREPARATION COMPLETE")
    print("=" * 70)

    print("\nNumerical tensors:")
    print(f"Train: {X_train_numeric.shape}")
    print(f"Val  : {X_val_numeric.shape}")
    print(f"Test : {X_test_numeric.shape}")

    print("\nCategorical tensors:")
    print(f"Train: {X_train_categorical.shape}")
    print(f"Val  : {X_val_categorical.shape}")
    print(f"Test : {X_test_categorical.shape}")

    print("\nLabels:")
    print(f"Train: {y_train.shape}")
    print(f"Val  : {y_val.shape}")
    print(f"Test : {y_test.shape}")

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()